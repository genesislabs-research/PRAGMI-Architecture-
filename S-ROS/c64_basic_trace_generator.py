"""
c64_basic_trace_generator.py
S-ROS Training Data Generator: C64 BASIC V2 Interpreter with State Traces

BIOLOGICAL GROUNDING
This file implements no neural dynamics. It is the ground-truth oracle that
produces deterministic execution traces for training the Theo procedural
execution core. Every trace captures the complete interpreter state before
and after each instruction, providing the supervised signal that Theo's
plastic side learns to predict. The program is always the ground truth.
The narrative is always a description of what happens in the world.

The trace format (JSONL, one JSON object per line) records:
    program_name: identifier for the source program
    state_before: full interpreter state prior to executing the current line
    state_after: full interpreter state after executing the current line

Interpreter state consists of:
    pc: current program counter (line number about to execute)
    vars: dictionary of variable name to float value
    gosub_stack: LIFO stack of return (line_number, sub_pc) tuples
    for_stack: stack of FOR loop contexts (var, limit, step, loop_line)
    data_pointer: index into the global DATA pool
    output: string produced by PRINT on this step (empty if none)
    halted: whether the program has terminated
    step_count: number of instructions executed so far

TRAINING METHODOLOGY
Eldan, R. and Li, Y. (2023). "TinyStories: How Small Can Language Models Be
and Still Speak Coherent English?" arXiv:2305.07759.
(Narrative-code interleaving methodology adapted for procedural traces.)

LANGUAGE REFERENCE
Kemeny, J.G. and Kurtz, T.E. (1964). "BASIC." Dartmouth College Computation
Center, 1 October 1964. Reproduced for non-commercial research use with
credit to Dartmouth College.

Genesis Labs Research, 2026
Amellia Mendel / LM Adler
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# =========================================================================
# INTERPRETER STATE
# =========================================================================

@dataclass
class InterpreterState:
    """Complete snapshot of the interpreter at one point in time.

    Every field is JSON-serializable so traces can be written directly
    to JSONL without transformation.

    Fields added in this revision:
        sub_pc: index of the current sub-statement within the colon-split
                list for the current line. Reset to 0 on any line change.
        _skip_rest_of_line: set True by _exec_if on a false branch so that
                step() advances to the next line rather than the next
                sub-statement. Cleared immediately after use. Not serialized.
    """

    pc: int = 0
    vars: Dict[str, float] = field(default_factory=dict)
    # gosub_stack entries are (return_line, return_sub_pc) tuples.
    # Serialized as lists of two-element lists for JSON compatibility.
    gosub_stack: List[Tuple[int, int]] = field(default_factory=list)
    for_stack: List[Dict[str, Any]] = field(default_factory=list)
    data_pointer: int = 0
    output: str = ""
    halted: bool = False
    step_count: int = 0
    sub_pc: int = 0
    _skip_rest_of_line: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pc": self.pc,
            "vars": dict(self.vars),
            # Serialize as list-of-lists for JSON; consumers reconstruct tuples.
            "gosub_stack": [list(entry) for entry in self.gosub_stack],
            "for_stack": [dict(f) for f in self.for_stack],
            "data_pointer": self.data_pointer,
            "output": self.output,
            "halted": self.halted,
            "step_count": self.step_count,
        }

    def clone(self) -> "InterpreterState":
        return InterpreterState(
            pc=self.pc,
            vars=dict(self.vars),
            gosub_stack=list(self.gosub_stack),
            for_stack=[dict(f) for f in self.for_stack],
            data_pointer=self.data_pointer,
            output=self.output,
            halted=self.halted,
            step_count=self.step_count,
            sub_pc=self.sub_pc,
            _skip_rest_of_line=self._skip_rest_of_line,
        )


# =========================================================================
# EXPRESSION EVALUATOR
# =========================================================================

class ExpressionEvaluator:
    """Evaluates BASIC arithmetic expressions with variables.

    Supports: +, -, *, /, ^, parentheses, unary minus, numeric literals,
    variable names (single letter or letter+digit), and the built-in
    functions INT, ABS, SQR, RND, PEEK.

    PEEK is evaluated via peek_handler if supplied; returns 0.0 otherwise
    (uninitialized C64 memory convention).

    NOT a biological component. Engineering utility for ground-truth
    trace generation only.
    """

    def __init__(
        self,
        variables: Dict[str, float],
        peek_handler: Optional[Callable[[int], int]] = None,
    ):
        self.variables = variables
        self.peek_handler = peek_handler
        self.pos = 0
        self.expr = ""

    def evaluate(self, expr: str) -> float:
        self.expr = expr.replace(" ", "")
        self.pos = 0
        result = self._parse_addition()
        return result

    def _peek(self) -> str:
        if self.pos < len(self.expr):
            return self.expr[self.pos]
        return ""

    def _consume(self, expected: str = "") -> str:
        ch = self._peek()
        if expected and ch != expected:
            raise SyntaxError(
                f"Expected '{expected}' at pos {self.pos} in '{self.expr}', "
                f"got '{ch}'"
            )
        self.pos += 1
        return ch

    def _parse_addition(self) -> float:
        left = self._parse_multiplication()
        while self._peek() in ("+", "-"):
            op = self._consume()
            right = self._parse_multiplication()
            if op == "+":
                left += right
            else:
                left -= right
        return left

    def _parse_multiplication(self) -> float:
        left = self._parse_power()
        while self._peek() in ("*", "/"):
            op = self._consume()
            right = self._parse_power()
            if op == "*":
                left *= right
            else:
                if right == 0:
                    raise ZeroDivisionError("Division by zero")
                left /= right
        return left

    def _parse_power(self) -> float:
        base = self._parse_unary()
        if self._peek() == "^":
            self._consume()
            exponent = self._parse_unary()
            return base ** exponent
        return base

    def _parse_unary(self) -> float:
        if self._peek() == "-":
            self._consume()
            return -self._parse_atom()
        if self._peek() == "+":
            self._consume()
        return self._parse_atom()

    def _parse_atom(self) -> float:
        # Parenthesized expression
        if self._peek() == "(":
            self._consume("(")
            val = self._parse_addition()
            self._consume(")")
            return val

        # PEEK must be checked before the generic built-in loop and before
        # the variable-name branch, because "peek" would otherwise be parsed
        # as four variable characters p-e-e-k consuming the wrong tokens.
        if self.expr[self.pos:].upper().startswith("PEEK"):
            self.pos += 4  # consume "peek" (4 chars)
            # Skip any spaces between PEEK and its opening paren
            while self.pos < len(self.expr) and self.expr[self.pos] == " ":
                self.pos += 1
            if self.pos >= len(self.expr) or self.expr[self.pos] != "(":
                raise SyntaxError("PEEK requires a parenthesized argument")
            self.pos += 1  # consume opening paren
            # Walk forward counting depth to find the matching close paren
            depth = 1
            start = self.pos
            while self.pos < len(self.expr) and depth > 0:
                if self.expr[self.pos] == "(":
                    depth += 1
                elif self.expr[self.pos] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                self.pos += 1
            arg_str = self.expr[start:self.pos]
            self.pos += 1  # consume closing paren
            sub_eval = ExpressionEvaluator(self.variables, peek_handler=self.peek_handler)
            addr = int(sub_eval.evaluate(arg_str))
            if self.peek_handler is not None:
                return float(self.peek_handler(addr))
            return 0.0  # fallback: uninitialized C64 memory reads as 0

        # Built-in functions (INT, ABS, SQR, RND)
        for func_name in ("INT", "ABS", "SQR", "RND"):
            if self.expr[self.pos:].upper().startswith(func_name + "("):
                self.pos += len(func_name)
                self._consume("(")
                arg = self._parse_addition()
                self._consume(")")
                if func_name == "INT":
                    return float(int(arg))
                elif func_name == "ABS":
                    return abs(arg)
                elif func_name == "SQR":
                    return math.sqrt(arg)
                elif func_name == "RND":
                    # Deterministic for trace generation: return 0.5 always.
                    # Real C64 RND uses a seed; reproducible traces require a
                    # fixed value.
                    return 0.5

        # Numeric literal
        if self._peek().isdigit() or self._peek() == ".":
            start = self.pos
            while self.pos < len(self.expr) and (
                self.expr[self.pos].isdigit() or self.expr[self.pos] == "."
            ):
                self.pos += 1
            return float(self.expr[start : self.pos])

        # Variable name (letter, optionally followed by a digit)
        if self._peek().isalpha():
            name = self._consume()
            while self.pos < len(self.expr) and self.expr[self.pos].isalnum():
                name += self._consume()
            return self.variables.get(name, 0.0)

        raise SyntaxError(
            f"Unexpected character '{self._peek()}' at pos {self.pos} "
            f"in '{self.expr}'"
        )


# =========================================================================
# CONDITION EVALUATOR
# =========================================================================

def evaluate_condition(
    condition_str: str,
    variables: Dict[str, float],
    peek_handler: Optional[Callable[[int], int]] = None,
) -> bool:
    """Evaluate a BASIC IF condition (supports =, <>, <, >, <=, >=, AND, OR).

    Args:
        condition_str: the condition text from the IF statement
        variables: current interpreter variable dict
        peek_handler: optional PEEK callback threaded into ExpressionEvaluator

    Returns:
        Boolean result of the condition.

    NOT a biological component. Engineering utility only.
    """
    upper = condition_str.upper()
    # OR has lower precedence than AND
    or_parts = re.split(r'\bOR\b', upper)
    if len(or_parts) > 1:
        return any(
            evaluate_condition(part, variables, peek_handler=peek_handler)
            for part in or_parts
        )
    and_parts = re.split(r'\bAND\b', upper)
    if len(and_parts) > 1:
        return all(
            evaluate_condition(part, variables, peek_handler=peek_handler)
            for part in and_parts
        )

    # Single comparison
    evaluator = ExpressionEvaluator(variables, peek_handler=peek_handler)
    for op in ("<=", ">=", "<>", "<", ">", "="):
        idx = condition_str.find(op)
        if idx != -1:
            left_str = condition_str[:idx].strip()
            right_str = condition_str[idx + len(op) :].strip()
            left = evaluator.evaluate(left_str)
            evaluator2 = ExpressionEvaluator(variables, peek_handler=peek_handler)
            right = evaluator2.evaluate(right_str)
            if op == "=":
                return left == right
            elif op == "<>":
                return left != right
            elif op == "<":
                return left < right
            elif op == ">":
                return left > right
            elif op == "<=":
                return left <= right
            elif op == ">=":
                return left >= right
    raise SyntaxError(f"No comparison operator found in condition: {condition_str}")


# =========================================================================
# C64 BASIC INTERPRETER
# =========================================================================

class C64BasicInterpreter:
    """Deterministic C64 BASIC V2 subset interpreter.

    Supported statements: LET, PRINT, INPUT (from DATA supply), IF/THEN,
    GOTO, FOR/NEXT, GOSUB/RETURN, READ/DATA, DIM, REM, END, POKE.

    The FOR/NEXT implementation uses a stack (not a single variable) to
    support nested loops correctly. GOSUB/RETURN uses a LIFO stack of
    (return_line, return_sub_pc) tuples so that mid-line GOSUB correctly
    resumes at the sub-statement following the GOSUB.

    INPUT is satisfied from a pre-supplied list of values rather than
    interactive stdin, enabling fully deterministic trace generation.

    POKE and PEEK delegate to caller-supplied handlers (poke_handler,
    peek_handler). When no handler is set, POKE stores into self._memory
    and PEEK reads from it, defaulting to 0 for unwritten addresses.
    This keeps VIC-II and CIA knowledge out of the interpreter entirely.
    """

    def __init__(
        self,
        program_text: str,
        input_values: Optional[List[float]] = None,
        max_steps: int = 10000,
    ):
        self.max_steps = max_steps
        self.input_values = input_values or []
        self.input_pointer = 0

        # Parse program into sorted line number -> statement mapping
        self.lines: Dict[int, str] = {}
        self.sorted_line_numbers: List[int] = []
        self.data_pool: List[float] = []

        # Cache of colon-split sub-statement lists, keyed by line number.
        # Populated lazily on first visit to each line.
        self._line_cache: Dict[int, List[str]] = {}

        # Fallback memory for POKE/PEEK when no handler is set.
        # Stores byte values (0-255) keyed by address integer.
        self._memory: Dict[int, int] = {}

        # Caller-supplied hardware hooks. pong_trainer.py assigns:
        #   self.interpreter.peek_handler = self.vic2.peek
        #   self.interpreter.poke_handler = self.vic2.poke
        # When no external peek_handler is set, _effective_peek_handler reads
        # from self._memory so POKE/PEEK round-trips work without a caller.
        self._external_peek_handler: Optional[Callable[[int], int]] = None
        self._external_poke_handler: Optional[Callable[[int, int], None]] = None

        self._parse_program(program_text)
        self._collect_data()

        # Initialize interpreter state
        self.state = InterpreterState()
        if self.sorted_line_numbers:
            self.state.pc = self.sorted_line_numbers[0]

    def _default_peek(self, addr: int) -> int:
        """Read from the fallback _memory dict; return 0 for unwritten addresses.

        Args:
            addr: C64 memory address integer

        Returns:
            Byte value at addr, or 0 if not previously written by POKE.
        """
        return self._memory.get(addr, 0)  # 0 matches uninitialized C64 RAM

    @property
    def peek_handler(self) -> Callable[[int], int]:
        """Return the active PEEK handler.

        External handler takes priority; falls back to _default_peek so that
        POKE/PEEK round-trips work with no caller configuration required.
        """
        return self._external_peek_handler if self._external_peek_handler is not None else self._default_peek

    @peek_handler.setter
    def peek_handler(self, handler: Optional[Callable[[int], int]]) -> None:
        """Set an external PEEK handler (e.g. vic2.peek from pong_trainer).

        Args:
            handler: callable taking an address int and returning a byte int,
                     or None to revert to the internal _memory fallback.
        """
        self._external_peek_handler = handler

    @property
    def poke_handler(self) -> Optional[Callable[[int, int], None]]:
        """Return the active POKE handler, or None to use _memory fallback.

        Returns:
            External POKE callable if set, otherwise None.
        """
        return self._external_poke_handler

    @poke_handler.setter
    def poke_handler(self, handler: Optional[Callable[[int, int], None]]) -> None:
        """Set an external POKE handler (e.g. vic2.poke from pong_trainer).

        Args:
            handler: callable taking (address int, value int) and returning None,
                     or None to revert to the internal _memory fallback.
        """
        self._external_poke_handler = handler

    def _parse_program(self, text: str):
        """Parse BASIC source into line_number -> statement mapping.

        Args:
            text: raw BASIC program source as a multi-line string

        Returns:
            None. Populates self.lines and self.sorted_line_numbers.
        """
        for raw_line in text.strip().split("\n"):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            match = re.match(r"^(\d+)\s+(.*)", raw_line)
            if match:
                line_num = int(match.group(1))
                statement = match.group(2).strip()
                # Remove trailing semicolons (C64 print continuation)
                if statement.endswith(";"):
                    statement = statement[:-1].strip()
                self.lines[line_num] = statement
        self.sorted_line_numbers = sorted(self.lines.keys())

    def _collect_data(self):
        """Pre-scan all DATA statements and build the data pool.

        Returns:
            None. Populates self.data_pool.
        """
        for ln in self.sorted_line_numbers:
            stmt = self.lines[ln].upper()
            if stmt.startswith("DATA"):
                data_str = self.lines[ln][4:].strip()
                for item in data_str.split(","):
                    item = item.strip()
                    try:
                        self.data_pool.append(float(item))
                    except ValueError:
                        self.data_pool.append(0.0)

    def _next_line(self, current: int) -> Optional[int]:
        """Return the next line number after current, or None if at end.

        Args:
            current: the current line number

        Returns:
            Next line number, or None if current is the last line.
        """
        idx = self.sorted_line_numbers.index(current)
        if idx + 1 < len(self.sorted_line_numbers):
            return self.sorted_line_numbers[idx + 1]
        return None

    def _find_line(self, target: int) -> int:
        """Find a line number, raising if it does not exist.

        Args:
            target: the BASIC line number to locate

        Returns:
            target, validated as present in self.lines.
        """
        if target in self.lines:
            return target
        raise RuntimeError(f"Line {target} does not exist")

    def _split_statements(self, raw_line: str) -> List[str]:
        """Split a BASIC line on colons that are outside quoted strings.

        A sub-statement beginning with REM (case-insensitive, after strip)
        consumes the rest of the line. Empty sub-statements from trailing
        or doubled colons are dropped. Leading and trailing whitespace on
        each returned sub-statement is stripped.

        Args:
            raw_line: the raw statement text for a single BASIC line
                      (line number already stripped by _parse_program)

        Returns:
            List of sub-statement strings in order of appearance.
        """
        parts: List[str] = []
        current: List[str] = []
        in_quotes = False

        i = 0
        seg_start = 0  # index in raw_line where the current segment began
        while i < len(raw_line):
            ch = raw_line[i]
            if ch == '"':
                in_quotes = not in_quotes
                current.append(ch)
            elif ch == ":" and not in_quotes:
                segment = "".join(current).strip()
                if segment:
                    parts.append(segment)
                    # REM terminates the line. Reconstruct the full REM text
                    # by taking everything from the start of this segment to
                    # the end of raw_line, then return immediately.
                    if re.match(r'^REM\b', segment.upper()):
                        # Find where this segment started in raw_line so we
                        # can grab the untruncated remainder.
                        seg_begin = raw_line.upper().find("REM", seg_start)
                        if seg_begin != -1:
                            parts[-1] = raw_line[seg_begin:].strip()
                        return parts
                current = []
                seg_start = i + 1
            else:
                current.append(ch)
            i += 1

        # Flush the final segment
        segment = "".join(current).strip()
        if segment:
            parts.append(segment)

        return parts

    def _find_top_level_comma(self, s: str) -> int:
        """Return the index of the first comma not inside parens or quotes.

        Used by _exec_poke to split the address and value expressions.

        Args:
            s: the argument string after the POKE keyword

        Returns:
            Index of the top-level comma, or -1 if none found.
        """
        depth = 0
        in_quotes = False
        for i, ch in enumerate(s):
            if ch == '"':
                in_quotes = not in_quotes
            elif not in_quotes:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                elif ch == "," and depth == 0:
                    return i
        return -1

    def step(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Execute one sub-statement and return (state_before, state_after).

        This is the core trace-producing method. Each call advances the
        interpreter by exactly one BASIC sub-statement (one element of the
        colon-split list for the current line) and returns the complete
        state snapshots for training data generation.

        PC advancement is the sole responsibility of this method's end
        block. No _exec_* method calls _advance_pc directly.
        """
        if self.state.halted:
            raise RuntimeError("Program has halted")

        # Capture state before execution
        state_before = self.state.clone()
        state_before.output = ""  # output belongs to the step that produces it

        # Clear output for this step
        self.state.output = ""

        line_num = self.state.pc
        if line_num not in self.lines:
            self.state.halted = True
            self.state.step_count += 1
            state_after = self.state.clone()
            return state_before.to_dict(), state_after.to_dict()

        # Build or retrieve the colon-split sub-statement list for this line.
        sub_stmts = self._line_cache.get(line_num)
        if sub_stmts is None:
            sub_stmts = self._split_statements(self.lines[line_num])
            self._line_cache[line_num] = sub_stmts

        # An empty line (e.g. a bare line number) advances immediately.
        if not sub_stmts:
            self._advance_pc()
            self.state.sub_pc = 0
            self.state.step_count += 1
            state_after = self.state.clone()
            return state_before.to_dict(), state_after.to_dict()

        raw_stmt = sub_stmts[self.state.sub_pc]
        upper_stmt = raw_stmt.upper().strip()

        # Record PC before dispatch so we can detect control-flow changes.
        pc_before = self.state.pc

        # Dispatch
        if upper_stmt.startswith("REM") or upper_stmt == "":
            pass  # REM is a no-op; PC advancement handled below

        elif upper_stmt.startswith("LET "):
            self._exec_let(raw_stmt[4:])

        elif upper_stmt.startswith("PRINT"):
            self._exec_print(raw_stmt[5:])

        elif upper_stmt.startswith("INPUT"):
            self._exec_input(raw_stmt[5:])

        elif upper_stmt.startswith("IF"):
            self._exec_if(raw_stmt[2:])

        elif upper_stmt.startswith("GOTO"):
            target = int(upper_stmt[4:].strip())
            self.state.pc = self._find_line(target)

        elif upper_stmt.startswith("GOSUB"):
            target = int(upper_stmt[5:].strip())
            # Resume sub_pc is the position after this GOSUB sub-statement.
            resume_sub_pc = self.state.sub_pc + 1
            if resume_sub_pc >= len(sub_stmts):
                # GOSUB is the last (or only) sub-statement on this line;
                # return address is the next line, sub_pc 0.
                next_ln = self._next_line(line_num)
                if next_ln is not None:
                    self.state.gosub_stack.append((next_ln, 0))
            else:
                # GOSUB is mid-line; return to same line at resume_sub_pc.
                self.state.gosub_stack.append((line_num, resume_sub_pc))
            self.state.pc = self._find_line(target)

        elif upper_stmt == "RETURN":
            if not self.state.gosub_stack:
                raise RuntimeError("RETURN without GOSUB")
            ret_line, ret_sub_pc = self.state.gosub_stack.pop()
            self.state.pc = ret_line
            # Restore sub_pc directly; the end-of-step block will see that
            # pc changed and set sub_pc to 0, so we override that here by
            # deferring the assignment via a flag approach. Instead, we use
            # a direct assignment after the end-of-step block by temporarily
            # storing the target in a local and applying it at the end.
            # See post-dispatch block below.
            self._gosub_return_sub_pc = ret_sub_pc

        elif upper_stmt.startswith("FOR"):
            self._exec_for(raw_stmt[3:])

        elif upper_stmt.startswith("NEXT"):
            self._exec_next(raw_stmt[4:])

        elif upper_stmt.startswith("READ"):
            self._exec_read(raw_stmt[4:])

        elif upper_stmt.startswith("DATA"):
            pass  # DATA lines are pre-scanned; skip at runtime

        elif upper_stmt.startswith("DIM"):
            pass  # DIM is a no-op; arrays not implemented

        elif upper_stmt == "END":
            self.state.halted = True

        elif upper_stmt.startswith("POKE"):
            self._exec_poke(raw_stmt[4:])

        elif "=" in raw_stmt and not upper_stmt.startswith("IF"):
            # Implicit LET (e.g., "A = 5" without the LET keyword)
            self._exec_let(raw_stmt)

        else:
            pass  # Unknown statement: skip

        # ---- PC advancement (sole authority) ----
        control_flow_taken = self.state.pc != pc_before

        # Handle RETURN's sub_pc restoration before generic logic.
        if hasattr(self, "_gosub_return_sub_pc"):
            ret_sub_pc = self._gosub_return_sub_pc
            del self._gosub_return_sub_pc
            self.state.sub_pc = ret_sub_pc
        elif self.state._skip_rest_of_line:
            # IF false branch: skip the rest of this line, go to next line.
            self.state._skip_rest_of_line = False
            self._advance_pc()
            self.state.sub_pc = 0
        elif control_flow_taken:
            # GOTO, GOSUB (pc already set), FOR jump, NEXT loop-back all
            # land here. sub_pc resets to 0 for the new line.
            self.state.sub_pc = 0
        else:
            # Normal sequential execution within or across lines.
            self.state.sub_pc += 1
            if self.state.sub_pc >= len(sub_stmts):
                self._advance_pc()
                self.state.sub_pc = 0

        self.state.step_count += 1
        state_after = self.state.clone()
        return state_before.to_dict(), state_after.to_dict()

    def _advance_pc(self):
        """Move PC to the next line in sequence.

        Returns:
            None. Mutates self.state.pc or sets self.state.halted.
        """
        next_ln = self._next_line(self.state.pc)
        if next_ln is not None:
            self.state.pc = next_ln
        else:
            self.state.halted = True

    def _exec_let(self, assignment_str: str):
        """Execute a LET (or implicit LET) assignment.

        Args:
            assignment_str: text after the LET keyword, e.g. "A = 5 + B"

        Returns:
            None. Mutates self.state.vars.
        """
        assignment_str = assignment_str.strip()
        eq_idx = assignment_str.find("=")
        if eq_idx == -1:
            raise SyntaxError(f"No '=' in assignment: {assignment_str}")
        var_name = assignment_str[:eq_idx].strip().upper()
        # Uppercase the expression so that bare variable names like "a+b"
        # resolve to the uppercased keys stored in self.state.vars.
        expr_str = assignment_str[eq_idx + 1 :].strip().upper()
        evaluator = ExpressionEvaluator(self.state.vars, peek_handler=self.peek_handler)
        value = evaluator.evaluate(expr_str)
        self.state.vars[var_name] = value

    def _exec_print(self, args_str: str):
        """Execute a PRINT statement.

        Args:
            args_str: text after the PRINT keyword

        Returns:
            None. Mutates self.state.output.
        """
        args_str = args_str.strip()
        if not args_str:
            self.state.output = ""
            return

        # Handle string literals in quotes
        if args_str.startswith('"'):
            end_quote = args_str.find('"', 1)
            if end_quote != -1:
                self.state.output = args_str[1:end_quote]
            else:
                self.state.output = args_str[1:]
        else:
            # Expression
            evaluator = ExpressionEvaluator(self.state.vars, peek_handler=self.peek_handler)
            try:
                value = evaluator.evaluate(args_str)
                # C64 BASIC prints numbers with no trailing zeros for integers
                if value == int(value):
                    self.state.output = str(int(value))
                else:
                    self.state.output = str(value)
            except Exception:
                self.state.output = args_str

    def _exec_input(self, args_str: str):
        """Execute an INPUT statement using pre-supplied values.

        Args:
            args_str: text after the INPUT keyword, optionally including
                      a prompt string followed by a semicolon

        Returns:
            None. Mutates self.state.vars.
        """
        var_name = args_str.strip().upper()
        # Strip prompt string if present (e.g., INPUT "NAME";A)
        if '"' in var_name:
            semi_idx = var_name.find(";")
            if semi_idx != -1:
                var_name = var_name[semi_idx + 1 :].strip()

        if self.input_pointer < len(self.input_values):
            value = float(self.input_values[self.input_pointer])
            self.input_pointer += 1
        else:
            value = 0.0

        self.state.vars[var_name] = value

    def _exec_if(self, if_body: str):
        """Execute an IF/THEN statement.

        On a false branch, sets self.state._skip_rest_of_line so that
        step() advances to the next line rather than the next sub-statement.
        This matches BASIC semantics: a false IF skips the entire rest of
        the current line, not just the THEN clause.

        Args:
            if_body: text after the IF keyword, e.g. "X > 5 THEN 100"

        Returns:
            None. Mutates self.state.pc or self.state._skip_rest_of_line.
        """
        upper_body = if_body.upper()
        then_idx = upper_body.find("THEN")
        if then_idx == -1:
            raise SyntaxError(f"IF without THEN: {if_body}")

        condition_str = if_body[:then_idx].strip()
        action_str = if_body[then_idx + 4 :].strip()

        result = evaluate_condition(
            condition_str, self.state.vars, peek_handler=self.peek_handler
        )

        if result:
            action_upper = action_str.upper().strip()
            try:
                target = int(action_upper)
                self.state.pc = self._find_line(target)
            except ValueError:
                if action_upper.startswith("GOTO"):
                    target = int(action_upper[4:].strip())
                    self.state.pc = self._find_line(target)
                elif action_upper.startswith("PRINT"):
                    self._exec_print(action_str[5:])
                elif action_upper == "END":
                    self.state.halted = True
                elif action_upper == "RETURN":
                    if self.state.gosub_stack:
                        ret_line, ret_sub_pc = self.state.gosub_stack.pop()
                        self.state.pc = ret_line
                        self._gosub_return_sub_pc = ret_sub_pc
                    else:
                        raise RuntimeError("RETURN without GOSUB")
                else:
                    if "=" in action_str:
                        self._exec_let(action_str)
                    # If no recognized action, fall through normally
        else:
            # False branch: skip the rest of this line entirely.
            self.state._skip_rest_of_line = True

    def _exec_for(self, for_body: str):
        """Execute a FOR statement. Pushes a context onto the for_stack.

        FOR I = 1 TO 10 STEP 2: the loop variable is set, and the context
        (limit, step, loop_line) is pushed. loop_line points to the FOR line
        itself, so NEXT can use _next_line(loop_line) to find the loop body.

        If re-entering the same FOR variable, the old context is removed
        first (C64 behavior).

        Args:
            for_body: text after the FOR keyword

        Returns:
            None. Mutates self.state.vars and self.state.for_stack.
        """
        upper = for_body.upper()

        eq_idx = upper.find("=")
        to_idx = upper.find("TO")
        step_idx = upper.find("STEP")

        var_name = for_body[:eq_idx].strip().upper()

        if step_idx != -1:
            start_expr = for_body[eq_idx + 1 : to_idx].strip()
            limit_expr = for_body[to_idx + 2 : step_idx].strip()
            step_expr = for_body[step_idx + 4 :].strip()
        else:
            start_expr = for_body[eq_idx + 1 : to_idx].strip()
            limit_expr = for_body[to_idx + 2 :].strip()
            step_expr = "1"

        evaluator = ExpressionEvaluator(self.state.vars, peek_handler=self.peek_handler)
        start_val = evaluator.evaluate(start_expr)
        limit_val = ExpressionEvaluator(self.state.vars, peek_handler=self.peek_handler).evaluate(limit_expr)
        step_val = ExpressionEvaluator(self.state.vars, peek_handler=self.peek_handler).evaluate(step_expr)

        self.state.vars[var_name] = start_val

        # Remove any existing context for this variable (re-entry behavior)
        self.state.for_stack = [
            ctx for ctx in self.state.for_stack if ctx["var"] != var_name
        ]

        self.state.for_stack.append({
            "var": var_name,
            "limit": limit_val,
            "step": step_val,
            "loop_line": self.state.pc,  # NEXT uses _next_line(loop_line)
        })

        # FOR falls through to the next line; control_flow_taken will be
        # False (pc unchanged), so step()'s end block advances normally.

    def _exec_next(self, next_body: str):
        """Execute a NEXT statement. Pops or continues the FOR context.

        NEXT I: increment I by step. If I has not passed the limit, jump
        back to the line after the FOR. If done, pop the context and
        continue. PC is set directly here; step() sees control_flow_taken
        and resets sub_pc to 0.

        Args:
            next_body: text after the NEXT keyword, containing the variable

        Returns:
            None. Mutates self.state.vars, self.state.for_stack, self.state.pc.
        """
        var_name = next_body.strip().upper()

        ctx_idx = None
        for i in range(len(self.state.for_stack) - 1, -1, -1):
            if self.state.for_stack[i]["var"] == var_name:
                ctx_idx = i
                break

        if ctx_idx is None:
            raise RuntimeError(f"NEXT {var_name} without matching FOR")

        ctx = self.state.for_stack[ctx_idx]

        current = self.state.vars[var_name]
        current += ctx["step"]
        self.state.vars[var_name] = current

        step = ctx["step"]
        limit = ctx["limit"]
        done = (step > 0 and current > limit) or (step < 0 and current < limit)

        if done:
            self.state.for_stack = self.state.for_stack[:ctx_idx]
            # Fall through: pc unchanged, step() end block advances normally.
        else:
            # Jump to the line after the FOR statement to re-execute loop body.
            loop_line = ctx["loop_line"]
            jump_target = self._next_line(loop_line)
            if jump_target is not None:
                self.state.pc = jump_target  # control_flow_taken -> sub_pc=0
            else:
                self.state.halted = True

    def _exec_read(self, read_body: str):
        """Execute a READ statement, consuming values from the DATA pool.

        Args:
            read_body: text after the READ keyword, comma-separated var names

        Returns:
            None. Mutates self.state.vars and self.state.data_pointer.
        """
        var_names = [v.strip().upper() for v in read_body.split(",")]
        for var_name in var_names:
            if self.state.data_pointer < len(self.data_pool):
                self.state.vars[var_name] = self.data_pool[self.state.data_pointer]
                self.state.data_pointer += 1
            else:
                self.state.vars[var_name] = 0.0

    def _exec_poke(self, args_str: str):
        """Execute a POKE address, value statement.

        Delegates to self.poke_handler if set; otherwise stores the value
        in self._memory clamped to the range 0-255 (C64 byte range).
        Both address and value are evaluated as full expressions.

        Args:
            args_str: text after the POKE keyword, e.g. "53280, 0"

        Returns:
            None. Calls poke_handler or mutates self._memory.
        """
        comma_idx = self._find_top_level_comma(args_str)
        if comma_idx == -1:
            raise SyntaxError(f"POKE requires address, value: {args_str}")
        addr_str = args_str[:comma_idx].strip()
        val_str = args_str[comma_idx + 1 :].strip()
        evaluator = ExpressionEvaluator(self.state.vars, peek_handler=self.peek_handler)
        addr = int(evaluator.evaluate(addr_str))
        evaluator2 = ExpressionEvaluator(self.state.vars, peek_handler=self.peek_handler)
        val = int(evaluator2.evaluate(val_str))
        if self.poke_handler is not None:
            self.poke_handler(addr, val)
        else:
            self._memory[addr] = max(0, min(255, val))  # clamp to C64 byte range

    def run(self) -> List[Tuple[Dict, Dict]]:
        """Run the program to completion, collecting all traces.

        Returns:
            List of (state_before, state_after) pairs, one per executed
            sub-statement.
        """
        traces = []
        while not self.state.halted and self.state.step_count < self.max_steps:
            before, after = self.step()
            traces.append((before, after))
        return traces


# =========================================================================
# TRACE WRITER
# =========================================================================

def write_traces(
    program_name: str,
    program_text: str,
    output_path: str,
    input_values: Optional[List[float]] = None,
    append: bool = True,
) -> int:
    """Execute a BASIC program and write traces to a JSONL file.

    Args:
        program_name: identifier for this program in the traces
        program_text: the BASIC source code
        output_path: path to the output JSONL file
        input_values: pre-supplied values for INPUT statements
        append: if True, append to existing file; if False, overwrite

    Returns:
        Number of trace pairs written
    """
    interp = C64BasicInterpreter(program_text, input_values=input_values)
    traces = interp.run()

    mode = "a" if append else "w"
    with open(output_path, mode) as f:
        for before, after in traces:
            record = {
                "program_name": program_name,
                "state_before": before,
                "state_after": after,
            }
            f.write(json.dumps(record) + "\n")

    return len(traces)


# =========================================================================
# TEST PROGRAMS
# =========================================================================

TEST_PROGRAMS = {
    "01_hello_world": {
        "code": """\
10 PRINT "HELLO WORLD"
20 END
""",
    },
    "02_simple_assignment": {
        "code": """\
10 LET A = 5
20 LET B = 3
30 PRINT A + B
40 END
""",
    },
    "03_count_to_five": {
        "code": """\
10 FOR I = 1 TO 5
20 PRINT I
30 NEXT I
40 END
""",
    },
    "04_sum_1_to_10": {
        "code": """\
10 LET S = 0
20 FOR I = 1 TO 10
30 S = S + I
40 NEXT I
50 PRINT S
60 END
""",
    },
    "05_nested_for": {
        "code": """\
10 FOR I = 1 TO 3
20 FOR J = 1 TO 2
30 PRINT I * 10 + J
40 NEXT J
50 NEXT I
60 END
""",
    },
    "06_if_then_goto": {
        "code": """\
10 LET X = 10
20 IF X > 5 THEN 40
30 PRINT "SMALL"
40 PRINT "BIG"
50 END
""",
    },
    "07_gosub_return": {
        "code": """\
10 GOSUB 100
20 PRINT "BACK"
30 END
100 PRINT "SUB"
110 RETURN
""",
    },
    "08_nested_gosub": {
        "code": """\
10 GOSUB 100
20 PRINT "DONE"
30 END
100 PRINT "OUTER"
110 GOSUB 200
120 RETURN
200 PRINT "INNER"
210 RETURN
""",
    },
    "09_read_data": {
        "code": """\
10 READ A
20 READ B
30 PRINT A + B
40 END
50 DATA 10, 25
""",
    },
    "10_while_loop": {
        "code": """\
10 LET X = 1
20 IF X > 5 THEN 50
30 X = X + 1
40 GOTO 20
50 PRINT X
60 END
""",
    },
    "11_factorial": {
        "code": """\
10 LET N = 5
20 LET F = 1
30 FOR I = 1 TO N
40 F = F * I
50 NEXT I
60 PRINT F
70 END
""",
    },
    "12_absolute_value": {
        "code": """\
10 LET X = -7
20 IF X < 0 THEN 40
30 GOTO 50
40 X = X * -1
50 PRINT X
60 END
""",
    },
    "13_for_step": {
        "code": """\
10 FOR I = 10 TO 1 STEP -2
20 PRINT I
30 NEXT I
40 END
""",
    },
    "14_string_output": {
        "code": """\
10 PRINT "GENESIS"
20 PRINT "LABS"
30 END
""",
    },
    "15_multiplication_table": {
        "code": """\
10 FOR I = 1 TO 3
20 FOR J = 1 TO 3
30 PRINT I * J
40 NEXT J
50 NEXT I
60 END
""",
    },
    "16_power_of_two": {
        "code": """\
10 LET P = 1
20 FOR I = 1 TO 8
30 P = P * 2
40 NEXT I
50 PRINT P
60 END
""",
    },
    "17_gosub_with_args": {
        "code": """\
10 LET X = 3
20 GOSUB 100
30 PRINT X
40 END
100 X = X * X
110 RETURN
""",
    },
    "18_countdown": {
        "code": """\
10 FOR I = 5 TO 0 STEP -1
20 IF I = 0 THEN 40
30 PRINT I
40 NEXT I
50 PRINT "GO"
60 END
""",
    },
    "19_rem_ignored": {
        "code": """\
10 REM This is a comment
20 LET X = 42
30 REM Another comment
40 PRINT X
50 END
""",
    },
    "20_multiple_inputs": {
        "code": """\
10 LET S = 0
20 FOR I = 1 TO 3
30 INPUT X
40 S = S + X
50 NEXT I
60 PRINT S
70 END
""",
        "inputs": [10, 20, 30],
    },
}


# =========================================================================
# SELF-TESTS
# =========================================================================

def run_tests() -> bool:
    """Run all test programs and verify correctness.

    Includes the original 28 tests (20 program runs + 8 correctness checks)
    plus 12 new tests for multi-statement lines, POKE, PEEK, and GOSUB
    mid-line behavior.

    Returns:
        True if all tests pass.
    """
    passed = 0
    failed = 0

    for name, spec in sorted(TEST_PROGRAMS.items()):
        code = spec["code"]
        inputs = spec.get("inputs", [])
        try:
            interp = C64BasicInterpreter(code, input_values=inputs)
            traces = interp.run()
            assert len(traces) > 0, "No traces produced"
            assert traces[-1][1]["halted"], "Program did not halt"
            print(f"  PASS: {name} ({len(traces)} steps)")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name}: {e}")
            failed += 1

    def _last_output(traces):
        for _, after in reversed(traces):
            if after["output"]:
                return after["output"]
        return ""

    correctness_tests = [
        ("03_count_to_five", lambda t: _last_output(t) == "5",
         "last printed value should be 5"),
        ("04_sum_1_to_10", lambda t: _last_output(t) == "55",
         "sum should be 55"),
        ("05_nested_for", lambda t: len(t) > 10,
         "nested loop should produce many steps"),
        ("07_gosub_return", lambda t: any(
            s[1]["output"] == "SUB" for s in t),
         "should print SUB"),
        ("11_factorial", lambda t: _last_output(t) == "120",
         "5! should be 120"),
        ("13_for_step", lambda t: t[0][1]["vars"].get("I") == 10.0,
         "FOR with negative step should start at 10"),
        ("16_power_of_two", lambda t: _last_output(t) == "256",
         "2^8 should be 256"),
        ("20_multiple_inputs", lambda t: _last_output(t) == "60",
         "sum of 10+20+30 should be 60"),
    ]

    for name, check_fn, description in correctness_tests:
        code = TEST_PROGRAMS[name]["code"]
        inputs = TEST_PROGRAMS[name].get("inputs", [])
        interp = C64BasicInterpreter(code, input_values=inputs)
        traces = interp.run()
        try:
            assert check_fn(traces), description
            print(f"  PASS: {name} correctness: {description}")
            passed += 1
        except AssertionError:
            print(f"  FAIL: {name} correctness: {description}")
            failed += 1
        except Exception as e:
            print(f"  FAIL: {name} correctness error: {e}")
            failed += 1

    # ---- New tests: colon-split, POKE, PEEK, GOSUB mid-line ----

    print("\n  -- New capability tests --")

    def _run(code, inputs=None, peek_handler=None, poke_handler=None):
        interp = C64BasicInterpreter(code, input_values=inputs or [])
        interp.peek_handler = peek_handler
        interp.poke_handler = poke_handler
        interp.run()
        return interp

    def _run_traces(code, inputs=None, peek_handler=None, poke_handler=None):
        interp = C64BasicInterpreter(code, input_values=inputs or [])
        interp.peek_handler = peek_handler
        interp.poke_handler = poke_handler
        traces = interp.run()
        return interp, traces

    # test_split_statements_simple
    try:
        interp = C64BasicInterpreter("10 END")
        result = interp._split_statements("a=1:b=2:c=3")
        assert result == ["a=1", "b=2", "c=3"], f"got {result}"
        print("  PASS: test_split_statements_simple")
        passed += 1
    except Exception as e:
        print(f"  FAIL: test_split_statements_simple: {e}")
        failed += 1

    # test_split_statements_quoted_colon
    try:
        interp = C64BasicInterpreter("10 END")
        result = interp._split_statements('print "a:b":x=1')
        assert result == ['print "a:b"', "x=1"], f"got {result}"
        print("  PASS: test_split_statements_quoted_colon")
        passed += 1
    except Exception as e:
        print(f"  FAIL: test_split_statements_quoted_colon: {e}")
        failed += 1

    # test_split_statements_rem_terminates
    try:
        interp = C64BasicInterpreter("10 END")
        result = interp._split_statements("a=1:rem b=2:c=3")
        assert result == ["a=1", "rem b=2:c=3"], f"got {result}"
        print("  PASS: test_split_statements_rem_terminates")
        passed += 1
    except Exception as e:
        print(f"  FAIL: test_split_statements_rem_terminates: {e}")
        failed += 1

    # test_split_statements_empty_drops
    try:
        interp = C64BasicInterpreter("10 END")
        result = interp._split_statements("a=1::b=2:")
        assert result == ["a=1", "b=2"], f"got {result}"
        print("  PASS: test_split_statements_empty_drops")
        passed += 1
    except Exception as e:
        print(f"  FAIL: test_split_statements_empty_drops: {e}")
        failed += 1

    # test_multi_statement_line_execution
    try:
        interp = _run("10 a=5:b=7:c=a+b\n20 end")
        v = interp.state.vars
        assert v.get("A") == 5.0 and v.get("B") == 7.0 and v.get("C") == 12.0, f"vars={v}"
        print("  PASS: test_multi_statement_line_execution")
        passed += 1
    except Exception as e:
        print(f"  FAIL: test_multi_statement_line_execution: {e}")
        failed += 1

    # test_poke_fallback_memory
    try:
        interp = _run("10 poke 1024, 42\n20 end")
        assert interp._memory.get(1024) == 42, f"memory={interp._memory}"
        print("  PASS: test_poke_fallback_memory")
        passed += 1
    except Exception as e:
        print(f"  FAIL: test_poke_fallback_memory: {e}")
        failed += 1

    # test_poke_handler_dispatch
    try:
        captured = []
        interp = _run("10 poke 1024, 42\n20 end", poke_handler=lambda a, v: captured.append((a, v)))
        assert captured == [(1024, 42)], f"captured={captured}"
        print("  PASS: test_poke_handler_dispatch")
        passed += 1
    except Exception as e:
        print(f"  FAIL: test_poke_handler_dispatch: {e}")
        failed += 1

    # test_peek_fallback_zero
    try:
        interp = _run("10 x = peek(1024)\n20 end")
        assert interp.state.vars.get("X") == 0.0, f"X={interp.state.vars.get('X')}"
        print("  PASS: test_peek_fallback_zero")
        passed += 1
    except Exception as e:
        print(f"  FAIL: test_peek_fallback_zero: {e}")
        failed += 1

    # test_peek_handler_dispatch
    try:
        interp = _run("10 x = peek(1024)\n20 end", peek_handler=lambda a: 99)
        assert interp.state.vars.get("X") == 99.0, f"X={interp.state.vars.get('X')}"
        print("  PASS: test_peek_handler_dispatch")
        passed += 1
    except Exception as e:
        print(f"  FAIL: test_peek_handler_dispatch: {e}")
        failed += 1

    # test_peek_in_compound (mirrors bpong.bas line 295 pattern)
    try:
        interp = _run("10 j=peek(100):k=peek(101)\n20 end", peek_handler=lambda a: a * 2)
        j = interp.state.vars.get("J")
        k = interp.state.vars.get("K")
        assert j == 200.0 and k == 202.0, f"J={j} K={k}"
        print("  PASS: test_peek_in_compound")
        passed += 1
    except Exception as e:
        print(f"  FAIL: test_peek_in_compound: {e}")
        failed += 1

    # test_poke_peek_round_trip (fallback _memory path)
    try:
        interp = _run("10 poke 100, 77\n20 x = peek(100)\n30 end")
        assert interp.state.vars.get("X") == 77.0, f"X={interp.state.vars.get('X')}"
        print("  PASS: test_poke_peek_round_trip")
        passed += 1
    except Exception as e:
        print(f"  FAIL: test_poke_peek_round_trip: {e}")
        failed += 1

    # test_gosub_mid_line
    # "10 a=1:gosub 100:b=2" -- GOSUB is mid-line; after RETURN, b=2 must execute.
    try:
        interp = _run("10 a=1:gosub 100:b=2\n20 end\n100 c=3:return")
        v = interp.state.vars
        assert v.get("A") == 1.0, f"A={v.get('A')}"
        assert v.get("B") == 2.0, f"B={v.get('B')}"
        assert v.get("C") == 3.0, f"C={v.get('C')}"
        print("  PASS: test_gosub_mid_line")
        passed += 1
    except Exception as e:
        print(f"  FAIL: test_gosub_mid_line: {e}")
        failed += 1

    # test_if_false_multistatement
    # BASIC semantics: false IF skips the rest of the line, not just THEN.
    # "10 if 1=2 then 100:a=5" -- a=5 must NOT execute on false branch.
    try:
        interp = _run("10 if 1=2 then 100:a=5\n20 end\n100 b=99")
        a_val = interp.state.vars.get("A")
        assert a_val is None or a_val == 0.0, \
            f"A should not be set by false IF branch, got {a_val}"
        print("  PASS: test_if_false_multistatement")
        passed += 1
    except Exception as e:
        print(f"  FAIL: test_if_false_multistatement: {e}")
        failed += 1

    total = passed + failed
    print(f"\n{passed}/{total} tests passed")
    return failed == 0


# =========================================================================
# CLI
# =========================================================================

def main():
    """Generate traces for all test programs, or run tests only."""
    if len(sys.argv) > 1 and sys.argv[1] == "--test-only":
        success = run_tests()
        sys.exit(0 if success else 1)

    output_path = "c64_basic_traces.jsonl"
    if len(sys.argv) > 1:
        output_path = sys.argv[1]

    # Clear output file
    open(output_path, "w").close()

    total_traces = 0
    for name, spec in sorted(TEST_PROGRAMS.items()):
        code = spec["code"]
        inputs = spec.get("inputs", [])
        count = write_traces(name, code, output_path, input_values=inputs)
        total_traces += count
        print(f"  {name}: {count} traces")

    print(f"\nTotal: {total_traces} traces written to {output_path}")


if __name__ == "__main__":
    main()
