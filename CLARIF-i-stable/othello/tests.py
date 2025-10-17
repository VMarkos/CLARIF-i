# tests.py

import unittest
from othello import OthelloGame

class TestOthelloGame(unittest.TestCase):
    """
    Unit tests for the OthelloGame class (the game logic component).
    """

    def setUp(self):
        """Set up a fresh OthelloGame instance before each test."""
        self.game = OthelloGame()

    def test_initialization(self):
        """Test the initial state of the board, player, and score."""
        expected_initial_board = [
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', 'W', 'B', '.', '.', '.'],  # (3,3)W, (3,4)B
            ['.', '.', '.', 'B', 'W', '.', '.', '.'],  # (4,3)B, (4,4)W
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.']
        ]
        
        # Check board state
        self.assertEqual(self.game.get_board(), expected_initial_board)
        
        # Check starting player
        self.assertEqual(self.game.get_current_player(), self.game.BLACK)
        
        # Check initial score
        self.assertEqual(self.game.get_score(), {self.game.BLACK: 2, self.game.WHITE: 2})

    def test_valid_moves_initial(self):
        """Test the valid moves for the starting player (Black)."""
        valid_moves = self.game.get_valid_moves()
        
        # Black's initial valid moves are C4, D3, E6, F5
        expected_moves = [(2, 3), (3, 2), (4, 5), (5, 4)]
        
        # Check that the move locations are correct
        self.assertCountEqual(valid_moves.keys(), expected_moves)
        
        # Spot check one move's flips (e.g., D3 (3,2) flips (3,3)W)
        self.assertIn((3, 3), valid_moves[(3, 2)])
        self.assertEqual(len(valid_moves[(3, 2)]), 1, "D3 should only flip one piece.")
        
        # Spot check another move's flips (e.g., E6 (4,5) flips (4,4)W)
        self.assertIn((4, 4), valid_moves[(4, 5)])
        self.assertEqual(len(valid_moves[(4, 5)]), 1, "E6 should only flip one piece.")


    def test_make_first_move(self):
        """Test making a valid move (D3) and the resulting board state."""
        # Black makes the move D3 (3, 2)
        self.assertTrue(self.game.make_move(3, 2))

        # Check the new state of the board at the moved location and flipped piece
        board = self.game.get_board()
        self.assertEqual(board[3][2], self.game.BLACK, "The new piece should be Black.")
        self.assertEqual(board[3][3], self.game.BLACK, "The flipped piece should be Black.")
        
        # Check the score update
        self.assertEqual(self.game.get_score(), {self.game.BLACK: 4, self.game.WHITE: 1})

        # Check the turn switch
        self.assertEqual(self.game.get_current_player(), self.game.WHITE)
        
    def test_invalid_move(self):
        """Test making an invalid move (e.g., onto an occupied square)."""
        # Try to place a piece on an already occupied square (3, 3) which is White
        self.assertFalse(self.game.make_move(3, 3))
        
        # Board and player should not have changed
        self.assertEqual(self.game.get_current_player(), self.game.BLACK)
        self.assertEqual(self.game.get_score(), {self.game.BLACK: 2, self.game.WHITE: 2})
        
    def test_forced_pass_and_game_over(self):
        """
        Set up a scenario where one player has no moves, forcing a pass, 
        and then a scenario where both players have no moves, ending the game.
        
        Note: This requires temporarily manipulating the board state for the test.
        """
        
        # Clear the board and set a specific state (e.g., to force White to have no moves)
        self.game.board = [[self.game.BLACK for _ in range(8)] for _ in range(8)]
        # Put one White piece where it can be surrounded, but make the surround impossible 
        self.game.board[0][0] = self.game.WHITE
        self.game.current_player = self.game.WHITE # It's White's turn

        # 1. Test Forced Pass
        self.assertFalse(self.game.get_valid_moves(self.game.WHITE), "White should have no moves.")
        
        # Since make_move requires a valid move, we trigger the pass/game over logic directly
        # by calling _update_turn (which is normally called after a move)
        self.game._update_turn()
        
        # After passing White, the turn should switch to Black
        self.assertEqual(self.game.get_current_player(), self.game.BLACK, "Turn should pass to Black.")
        self.assertFalse(self.game.game_over, "Game should not be over yet.")

        # 2. Test Game Over
        # Now, imagine Black also has no moves
        self.game.board[0][1] = self.game.WHITE # Give Black one spot to play that is now invalid
        self.game.board[0][0] = self.game.BLACK # Ensure 0,0 is Black to prevent flips
        
        self.game.current_player = self.game.BLACK # Ensure it's black's turn
        
        # Manually clear all valid moves for Black as well
        self.game.board = [[self.game.BLACK for _ in range(8)] for _ in range(8)]
        self.game.board[7][7] = self.game.WHITE # Isolated piece, no moves for either

        # Set player to Black, so it tries to play, realizes it can't, passes to White
        self.game.current_player = self.game.BLACK
        
        self.game._update_turn()
        
        self.assertTrue(self.game.game_over, "Game should be over when both players have no moves.")
        self.assertEqual(self.game.determine_winner(), "B Wins! (63 to 1)", "Black should win.")
        
    def test_determine_winner_tie(self):
        """Test the tie scenario."""
        # Manually set a tie score state
        self.game.board = [[self.game.BLACK] * 4 + [self.game.WHITE] * 4] * 8
        self.game.board[0][0] = self.game.EMPTY # Not strictly necessary, but ensuring piece count is not 64
        self.assertEqual(self.game.determine_winner(), "It's a Tie! (32 to 32)")

# --- How to Run the Tests ---
if __name__ == '__main__':
    # You can customize the verbosity here.
    # unittest.main() runs all tests in the module.
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
