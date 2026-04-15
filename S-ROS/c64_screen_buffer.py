"""
c64_screen_buffer.py
Simulates the Commodore 64's 40x25 text mode screen buffer.

The C64 text screen is 40 columns wide and 25 rows tall, giving a
1000-character buffer. Characters are stored in screen RAM starting
at address $0400. This class models that buffer and the cursor
behavior: wrapping at column 40, scrolling when the cursor moves
past row 24.
"""

from typing import List


class C64ScreenBuffer:
    """
    Commodore 64 40x25 text mode screen buffer.

    Models screen RAM behavior: 1000-character buffer, cursor tracking,
    line wrap at column 40, and upward scroll when the cursor moves
    past row 24. Initialized to spaces, matching C64 power-on state.
    """

    COLS = 40
    ROWS = 25

    def __init__(self):
        """Initialize buffer to spaces with cursor at (0, 0)."""
        self._buffer: List[List[str]] = [
            [" "] * self.COLS for _ in range(self.ROWS)
        ]
        self._row = 0
        self._col = 0

    def _scroll(self):
        """
        Scroll the screen up by one row. The top row is discarded,
        a new blank row appears at the bottom, and the cursor moves
        to the start of the last row.
        """
        self._buffer.pop(0)
        self._buffer.append([" "] * self.COLS)
        self._row = self.ROWS - 1
        self._col = 0

    def print_string(self, s: str):
        """
        Write characters at the current cursor position, advancing
        the cursor after each character. Wraps to the next row at
        column 40. Scrolls when the cursor moves past row 24.

        Args:
            s: String to write to the screen buffer.
        """
        for ch in s:
            self._buffer[self._row][self._col] = ch
            self._col += 1
            if self._col >= self.COLS:
                self._col = 0
                self._row += 1
                if self._row >= self.ROWS:
                    self._scroll()

    def print_newline(self):
        """
        Move cursor to the start of the next row.
        Scrolls if already on the last row.
        """
        self._col = 0
        self._row += 1
        if self._row >= self.ROWS:
            self._scroll()

    def get_screen(self) -> List[str]:
        """
        Return the current screen contents as a list of 25 strings,
        each exactly 40 characters wide.

        Returns:
            List of 25 strings representing each row of the screen.
        """
        return ["".join(row) for row in self._buffer]

    @property
    def cursor_row(self) -> int:
        """Current cursor row (0-indexed)."""
        return self._row

    @property
    def cursor_col(self) -> int:
        """Current cursor column (0-indexed)."""
        return self._col


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def run_test():
    """
    Print 'HELLO WORLD', force scrolling with 30 newlines, and verify
    the final screen state.
    """
    screen = C64ScreenBuffer()

    # Print the string on row 0
    screen.print_string("HELLO WORLD")
    assert screen.get_screen()[0].startswith("HELLO WORLD"), \
        "HELLO WORLD not on row 0"
    print("PASS: HELLO WORLD written to row 0")

    # Force 30 newlines. Screen is 25 rows so this scrolls 6 times.
    # After scrolling, HELLO WORLD should have scrolled off the top.
    for _ in range(30):
        screen.print_newline()

    rows = screen.get_screen()

    # Verify exactly 25 rows returned
    assert len(rows) == 25, f"Expected 25 rows, got {len(rows)}"
    print("PASS: Screen has exactly 25 rows")

    # Verify each row is exactly 40 characters
    for i, row in enumerate(rows):
        assert len(row) == 40, f"Row {i} is {len(row)} chars, expected 40"
    print("PASS: All rows are exactly 40 characters")

    # HELLO WORLD should have scrolled off the top entirely
    full_text = "".join(rows)
    assert "HELLO WORLD" not in full_text, \
        "HELLO WORLD should have scrolled off screen"
    print("PASS: HELLO WORLD scrolled off after 30 newlines")

    # Cursor should be on the last row after scrolling
    assert screen.cursor_row == 24, \
        f"Expected cursor on row 24, got {screen.cursor_row}"
    assert screen.cursor_col == 0, \
        f"Expected cursor on col 0, got {screen.cursor_col}"
    print("PASS: Cursor at (24, 0) after scrolling")

    # Verify the screen is otherwise blank (all spaces)
    assert full_text == " " * 1000, \
        "Screen should be all spaces after scrolling past content"
    print("PASS: Screen is blank after content scrolled off")

    print("\nAll C64ScreenBuffer tests passed.")


if __name__ == "__main__":
    run_test()
