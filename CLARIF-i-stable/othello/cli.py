# cli.py

from othello import OthelloGame

class OthelloCLI:
    def __init__(self):
        """Initializes the CLI game instance."""
        self.game = OthelloGame()

    def _display_board(self, valid_moves=None):
        """Prints the current board state to the console."""
        board = self.game.get_board()
        size = self.game.BOARD_SIZE

        # Print column headers (A, B, C...)
        header = "   " + " ".join([chr(ord('A') + i) for i in range(size)])
        print(header)

        for r in range(size):
            # Row number (1, 2, 3...)
            row_display = [str(r + 1)]

            for c in range(size):
                piece = board[r][c]
                # Highlight valid moves with a '*'
                if valid_moves and (r, c) in valid_moves:
                    row_display.append('*')
                else:
                    row_display.append(piece)

            print(" " + " ".join(row_display))

    def _parse_input(self, move_str):
        """
        Parses a move string like 'A1' into (row, col) coordinates.
        Returns (r, c) tuple or None if invalid.
        """
        move_str = move_str.strip().upper()
        if len(move_str) != 2:
            return None

        col_char = move_str[0]
        row_char = move_str[1]

        # Convert column character (A->0, B->1...)
        col = ord(col_char) - ord('A')
        # Convert row character (1->0, 2->1...)
        try:
            row = int(row_char) - 1
        except ValueError:
            return None

        if 0 <= row < self.game.BOARD_SIZE and 0 <= col < self.game.BOARD_SIZE:
            return row, col
        else:
            return None

    def run(self):
        """The main game loop."""
        print("--- Othello (Reversi) CLI ---")
        print("Black is 'B', White is 'W'.")
        print("To make a move, enter coordinates like 'C4' (Column then Row).")
        print("Valid moves are marked with a '*' on the board.")
        print("Enter 'quit' to exit.")

        while not self.game.game_over:
            current_player = self.game.get_current_player()
            valid_moves = self.game.get_valid_moves(current_player)
            score = self.game.get_score()

            print(f"\n--- Current Player: {current_player} ---")
            print(f"Score: B={score[self.game.BLACK]} W={score[self.game.WHITE]}")

            self._display_board(valid_moves)

            if not valid_moves:
                # This should be handled internally by _update_turn, but good for feedback
                print(f"Player {current_player} has no moves and must pass. Checking game state...")
                self.game._update_turn() # Force a check/pass (if not already handled by a previous move)
                continue

            user_input = input("Enter your move (e.g., A1) or 'quit': ").strip()

            if user_input.lower() == 'quit':
                print("Exiting game.")
                break

            coords = self._parse_input(user_input)
            
            if coords is None:
                print("Invalid input format. Please use format like 'A1'.")
                continue

            row, col = coords
            
            if not self.game.make_move(row, col):
                print(f"Invalid move at {user_input}. Please choose a square marked with '*'.")

        if self.game.game_over:
            print("\n====================")
            print("||   GAME OVER!   ||")
            print("====================")
            print(self.game.determine_winner())
            score = self.game.get_score()
            print(f"Final Score: B={score[self.game.BLACK]} W={score[self.game.WHITE]}")


if __name__ == "__main__":
    cli = OthelloCLI()
    cli.run()

