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
    gosub_stack: LIFO stack of return line numbers (for GOSUB/RETURN)
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
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# =========================================================================
# INTERPRETER STATE
# =========================================================================

@dataclass
class InterpreterState:
    """Complete snapshot of the interpreter at one point in time.

    Every field is JSON-serializable so traces can be written directly
    to JSONL without transformation.
    """

    pc: int = 0
    vars: Dict[str, float] = field(default_factory=dict)
    gosub_stack: List[int] = field(default_factory=list)
    for_stack: List[Dict[str, Any]] = field(default_factory=list)
    data_pointer: int = 0
    output: str = ""
    halted: bool = False
    step_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pc": self.pc,
            "vars": dict(self.vars),
            "gosub_stack": list(self.gosub_stack),
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
        )


# =========================================================================
# EXPRESSION EVALUATOR
# =========================================================================

class ExpressionEvaluator:
    """Evaluates BASIC arithmetic expressions with variables.

    Supports: +, -, *, /, ^, parentheses, unary minus, numeric literals,
    variable names (single letter or letter+digit), and the built-in
    functions INT, ABS, SQR, RND.

    NOT a biological component. Engineering utility for ground-truth
    trace generation only.
    """

    def __init__(self, variables: Dict[str, float]):
        self.variables = variables
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

        # Built-in functions
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
                    # Deterministic for trace generation: return 0.5 always
                    # Real C64 RND uses a seed; for training data we want
                    # reproducible traces.
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

def evaluate_condition(condition_str: str, variables: Dict[str, float]) -> bool:
    """Evaluate a BASIC IF condition (supports =, <>, <, >, <=, >=, AND, OR).

    NOT a biological component. Engineering utility only.
    """
    # Handle AND/OR by splitting recursively
    upper = condition_str.upper()
    # OR has lower precedence
    or_parts = re.split(r'\bOR\b', upper)
    if len(or_parts) > 1:
        return any(
            evaluate_condition(part, variables) for part in or_parts
        )
    # AND
    and_parts = re.split(r'\bAND\b', upper)
    if len(and_parts) > 1:
        return all(
            evaluate_condition(part, variables) for part in and_parts
        )

    # Single comparison
    evaluator = ExpressionEvaluator(variables)
    for op in ("<=", ">=", "<>", "<", ">", "="):
        idx = condition_str.find(op)
        if idx != -1:
            left_str = condition_str[:idx].strip()
            right_str = condition_str[idx + len(op) :].strip()
            left = evaluator.evaluate(left_str)
            evaluator2 = ExpressionEvaluator(variables)
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
    GOTO, FOR/NEXT, GOSUB/RETURN, READ/DATA, DIM, REM, END.

    The FOR/NEXT implementation uses a stack (not a single variable) to
    support nested loops correctly. GOSUB/RETURN uses a LIFO stack.

    INPUT is satisfied from a pre-supplied list of values rather than
    interactive stdin, enabling fully deterministic trace generation.
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

        self._parse_program(program_text)
        self._collect_data()

        # Initialize interpreter state
        self.state = InterpreterState()
        if self.sorted_line_numbers:
            self.state.pc = self.sorted_line_numbers[0]

    def _parse_program(self, text: str):
        """Parse BASIC source into line_number -> statement mapping."""
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
        """Pre-scan all DATA statements and build the data pool."""
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
        """Return the next line number after current, or None if at end."""
        idx = self.sorted_line_numbers.index(current)
        if idx + 1 < len(self.sorted_line_numbers):
            return self.sorted_line_numbers[idx + 1]
        return None

    def _find_line(self, target: int) -> int:
        """Find a line number, raising if it does not exist."""
        if target in self.lines:
            return target
        raise RuntimeError(f"Line {target} does not exist")

    def step(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Execute one statement and return (state_before, state_after).

        This is the core trace-producing method. Each call advances the
        interpreter by exactly one BASIC statement and returns the complete
        state snapshots for training data generation.
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
            state_after = self.state.clone()
            self.state.step_count += 1
            return state_before.to_dict(), state_after.to_dict()

        raw_stmt = self.lines[line_num]
        upper_stmt = raw_stmt.upper().strip()

        # Dispatch
        if upper_stmt.startswith("REM") or upper_stmt == "":
            self._advance_pc()

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
            next_ln = self._next_line(line_num)
            if next_ln is not None:
                self.state.gosub_stack.append(next_ln)
            self.state.pc = self._find_line(target)

        elif upper_stmt == "RETURN":
            if not self.state.gosub_stack:
                raise RuntimeError("RETURN without GOSUB")
            self.state.pc = self.state.gosub_stack.pop()

        elif upper_stmt.startswith("FOR"):
            self._exec_for(raw_stmt[3:])

        elif upper_stmt.startswith("NEXT"):
            self._exec_next(raw_stmt[4:])

        elif upper_stmt.startswith("READ"):
            self._exec_read(raw_stmt[4:])

        elif upper_stmt.startswith("DATA"):
            # DATA lines are pre-scanned; skip at runtime
            self._advance_pc()

        elif upper_stmt.startswith("DIM"):
            # DIM is a no-op in this interpreter (arrays not implemented)
            self._advance_pc()

        elif upper_stmt == "END":
            self.state.halted = True

        elif "=" in raw_stmt and not upper_stmt.startswith("IF"):
            # Implicit LET (e.g., "A = 5" without the LET keyword)
            self._exec_let(raw_stmt)

        else:
            # Unknown statement: skip
            self._advance_pc()

        self.state.step_count += 1

        state_after = self.state.clone()
        return state_before.to_dict(), state_after.to_dict()

    def _advance_pc(self):
        """Move PC to the next line in sequence."""
        next_ln = self._next_line(self.state.pc)
        if next_ln is not None:
            self.state.pc = next_ln
        else:
            self.state.halted = True

    def _exec_let(self, assignment_str: str):
        """Execute a LET (or implicit LET) assignment."""
        assignment_str = assignment_str.strip()
        eq_idx = assignment_str.find("=")
        if eq_idx == -1:
            raise SyntaxError(f"No '=' in assignment: {assignment_str}")
        var_name = assignment_str[:eq_idx].strip().upper()
        expr_str = assignment_str[eq_idx + 1 :].strip()
        evaluator = ExpressionEvaluator(self.state.vars)
        value = evaluator.evaluate(expr_str)
        self.state.vars[var_name] = value
        self._advance_pc()

    def _exec_print(self, args_str: str):
        """Execute a PRINT statement."""
        args_str = args_str.strip()
        if not args_str:
            self.state.output = ""
            self._advance_pc()
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
            evaluator = ExpressionEvaluator(self.state.vars)
            try:
                value = evaluator.evaluate(args_str)
                # C64 BASIC prints numbers with no trailing zeros for integers
                if value == int(value):
                    self.state.output = str(int(value))
                else:
                    self.state.output = str(value)
            except Exception:
                self.state.output = args_str

        self._advance_pc()

    def _exec_input(self, args_str: str):
        """Execute an INPUT statement using pre-supplied values."""
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
        self._advance_pc()

    def _exec_if(self, if_body: str):
        """Execute an IF/THEN statement."""
        upper_body = if_body.upper()
        then_idx = upper_body.find("THEN")
        if then_idx == -1:
            raise SyntaxError(f"IF without THEN: {if_body}")

        condition_str = if_body[:then_idx].strip()
        action_str = if_body[then_idx + 4 :].strip()

        result = evaluate_condition(condition_str, self.state.vars)

        if result:
            # Action is either a line number (GOTO) or a statement
            action_upper = action_str.upper().strip()
            try:
                target = int(action_upper)
                self.state.pc = self._find_line(target)
            except ValueError:
                # Inline statement after THEN (e.g., IF X>0 THEN PRINT X)
                if action_upper.startswith("GOTO"):
                    target = int(action_upper[4:].strip())
                    self.state.pc = self._find_line(target)
                elif action_upper.startswith("PRINT"):
                    self._exec_print(action_str[5:])
                elif action_upper == "END":
                    self.state.halted = True
                elif action_upper == "RETURN":
                    if self.state.gosub_stack:
                        self.state.pc = self.state.gosub_stack.pop()
                    else:
                        raise RuntimeError("RETURN without GOSUB")
                else:
                    # Try as implicit LET
                    if "=" in action_str:
                        self._exec_let(action_str)
                    else:
                        self._advance_pc()
        else:
            self._advance_pc()

    def _exec_for(self, for_body: str):
        """Execute a FOR statement. Pushes a context onto the for_stack.

        FOR I = 1 TO 10 STEP 2
        The loop variable is set, and the context (limit, step, loop_line)
        is pushed. loop_line points to the line AFTER the FOR statement,
        which is where NEXT will jump back to.

        If re-entering the same FOR variable, the old context is removed
        first (C64 behavior: re-entering a FOR clears the previous context
        for that variable).
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

        evaluator = ExpressionEvaluator(self.state.vars)
        start_val = evaluator.evaluate(start_expr)
        limit_val = ExpressionEvaluator(self.state.vars).evaluate(limit_expr)
        step_val = ExpressionEvaluator(self.state.vars).evaluate(step_expr)

        # Set the loop variable
        self.state.vars[var_name] = start_val

        # Remove any existing context for this variable (re-entry behavior)
        self.state.for_stack = [
            ctx for ctx in self.state.for_stack if ctx["var"] != var_name
        ]

        # The loop_line is the line AFTER the FOR, which is where NEXT
        # will send execution to continue the loop body.
        next_line = self._next_line(self.state.pc)

        self.state.for_stack.append({
            "var": var_name,
            "limit": limit_val,
            "step": step_val,
            "loop_line": self.state.pc,
        })

        self._advance_pc()

    def _exec_next(self, next_body: str):
        """Execute a NEXT statement. Pops or continues the FOR context.

        NEXT I: increment I by step. If I has not passed the limit,
        jump back to the line after the FOR. If I has passed the limit,
        pop the context and continue to the next line.

        Uses a stack so nested FOR loops work correctly.
        """
        var_name = next_body.strip().upper()

        # Find the matching FOR context (search from top of stack)
        ctx_idx = None
        for i in range(len(self.state.for_stack) - 1, -1, -1):
            if self.state.for_stack[i]["var"] == var_name:
                ctx_idx = i
                break

        if ctx_idx is None:
            raise RuntimeError(f"NEXT {var_name} without matching FOR")

        ctx = self.state.for_stack[ctx_idx]

        # Increment the loop variable
        current = self.state.vars[var_name]
        current += ctx["step"]
        self.state.vars[var_name] = current

        # Check termination
        step = ctx["step"]
        limit = ctx["limit"]
        done = False
        if step > 0 and current > limit:
            done = True
        elif step < 0 and current < limit:
            done = True

        if done:
            # Remove this context and all contexts above it (C64 behavior)
            self.state.for_stack = self.state.for_stack[:ctx_idx]
            self._advance_pc()
        else:
            # Jump to the line after the FOR statement
            loop_line = ctx["loop_line"]
            jump_target = self._next_line(loop_line)
            if jump_target is not None:
                self.state.pc = jump_target
            else:
                self.state.halted = True

    def _exec_read(self, read_body: str):
        """Execute a READ statement, consuming values from the DATA pool."""
        var_names = [v.strip().upper() for v in read_body.split(",")]
        for var_name in var_names:
            if self.state.data_pointer < len(self.data_pool):
                self.state.vars[var_name] = self.data_pool[
                    self.state.data_pointer
                ]
                self.state.data_pointer += 1
            else:
                self.state.vars[var_name] = 0.0
        self._advance_pc()

    def run(self) -> List[Tuple[Dict, Dict]]:
        """Run the program to completion, collecting all traces.

        Returns a list of (state_before, state_after) pairs, one per
        executed statement.
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
    """Run all 20 test programs and verify basic correctness.

    Returns True if all tests pass.
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

    # Specific correctness checks
    def _last_output(traces):
        """Find the last non-empty output in a trace sequence."""
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
