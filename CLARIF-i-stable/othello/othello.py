# othello.py

class OthelloGame:
    # Constants for board size and piece representation
    BOARD_SIZE = 8
    EMPTY = '.'
    BLACK = 'B'
    WHITE = 'W'

    def __init__(self):
        """Initializes a new Othello game."""
        # Initialize an 8x8 board with empty spaces
        self.board = [[self.EMPTY for _ in range(self.BOARD_SIZE)] for _ in range(self.BOARD_SIZE)]
        # Set up the initial four pieces
        self.board[3][3] = self.WHITE
        self.board[3][4] = self.BLACK
        self.board[4][3] = self.BLACK
        self.board[4][4] = self.WHITE

        self.current_player = self.BLACK  # Black always starts
        self.game_over = False

    def get_board(self):
        """Returns the current state of the board."""
        return self.board

    def get_current_player(self):
        """Returns the piece of the current player."""
        return self.current_player

    def _get_opponent(self, player):
        """Helper to get the opponent's piece."""
        return self.WHITE if player == self.BLACK else self.BLACK

    def _is_on_board(self, r, c):
        """Helper to check if coordinates are within the board boundaries."""
        return 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE

    def get_valid_moves(self, player=None):
        """
        Calculates and returns a dictionary of valid moves for a player.
        Key: (row, col) tuple of the move location.
        Value: List of (r, c) tuples of pieces to be flipped.
        """
        if player is None:
            player = self.current_player

        valid_moves = {}
        opponent = self._get_opponent(player)

        # Iterate over every square on the board
        for r_start in range(self.BOARD_SIZE):
            for c_start in range(self.BOARD_SIZE):
                # A move must be on an empty square
                if self.board[r_start][c_start] == self.EMPTY:
                    flips = []
                    # Check all 8 directions
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue  # Skip the 'no direction' case

                            r, c = r_start + dr, c_start + dc
                            line_to_flip = []
                            # 1. Must be adjacent to an opponent's piece
                            if self._is_on_board(r, c) and self.board[r][c] == opponent:
                                line_to_flip.append((r, c))
                                # 2. Continue in the same direction
                                r, c = r + dr, c + dc
                                while self._is_on_board(r, c) and self.board[r][c] == opponent:
                                    line_to_flip.append((r, c))
                                    r, c = r + dr, c + dc

                                # 3. Must end on the current player's piece
                                if self._is_on_board(r, c) and self.board[r][c] == player:
                                    flips.extend(line_to_flip)

                    # If we found flips, then (r_start, c_start) is a valid move
                    if flips:
                        valid_moves[(r_start, c_start)] = flips

        return valid_moves

    def make_move(self, r, c):
        """
        Attempts to make a move at (r, c).
        Returns True if successful, False otherwise.
        Updates the board and flips the pieces.
        """
        if self.game_over:
            print("Game is over.")
            return False

        valid_moves = self.get_valid_moves(self.current_player)

        if (r, c) in valid_moves:
            # 1. Place the piece
            self.board[r][c] = self.current_player

            # 2. Flip the pieces
            for flip_r, flip_c in valid_moves[(r, c)]:
                self.board[flip_r][flip_c] = self.current_player

            # 3. Check for next player's move and potentially pass
            self._update_turn()
            return True
        else:
            return False

    def _update_turn(self):
        """
        Switches the current player and handles pass/game over logic.
        """
        opponent = self._get_opponent(self.current_player)
        self.current_player = opponent  # Tentatively switch

        # Check if the new player has any moves
        if not self.get_valid_moves(self.current_player):
            print(f"Player {self.current_player} has no valid moves. Passing turn back...")
            
            # Switch back to the previous player
            self.current_player = self._get_opponent(self.current_player)

            # Check if the original player (who just played) also has no moves
            if not self.get_valid_moves(self.current_player):
                print("Both players have no valid moves. Game Over!")
                self.game_over = True
                self.determine_winner()

    def get_score(self):
        """Calculates the score for Black and White."""
        score = {self.BLACK: 0, self.WHITE: 0}
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                if self.board[r][c] == self.BLACK:
                    score[self.BLACK] += 1
                elif self.board[r][c] == self.WHITE:
                    score[self.WHITE] += 1
        return score

    def determine_winner(self):
        """Determines and returns the winner based on the current score."""
        scores = self.get_score()
        black_score = scores[self.BLACK]
        white_score = scores[self.WHITE]

        if black_score > white_score:
            return f"{self.BLACK} Wins! ({black_score} to {white_score})"
        elif white_score > black_score:
            return f"{self.WHITE} Wins! ({white_score} to {black_score})"
        else:
            return f"It's a Tie! ({black_score} to {white_score})"

