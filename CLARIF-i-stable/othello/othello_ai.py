# othello_ai.py

from othello import OthelloGame
from minimax_search import AbstractState, AlphaBetaPruning
import copy

class OthelloAIState(AbstractState):
    """
    A wrapper class for OthelloGame that implements the AbstractState interface.
    """
    
    # Store the OthelloGame constants for convenience
    BLACK = OthelloGame.BLACK
    WHITE = OthelloGame.WHITE
    
    # Define a simple, dummy heuristic value
    DUMMY_HEURISTIC_VALUE = 100

    def __init__(self, game: OthelloGame):
        """Initializes the state with a copy of the Othello game object."""
        # Use a deep copy to ensure the search explores new, isolated states
        self.game = copy.deepcopy(game)
        
    def is_terminal(self) -> bool:
        """Check if the game is over."""
        # Note: We must also check if the current player has moves, as the game
        # logic handles the double-pass game end *during* a move.
        return self.game.game_over or not self.game.get_valid_moves()

    def get_moves(self):
        """Returns a list of valid moves (r, c) tuples."""
        # Returns a list of the (r, c) tuples that represent valid moves
        return list(self.game.get_valid_moves().keys())

    def make_move(self, move):
        """
        Applies a move and returns a *new* OthelloAIState object.
        'move' is an (r, c) tuple.
        """
        r, c = move
        
        # Create a new game instance for the next state
        new_game = copy.deepcopy(self.game)
        
        # Make the move (this mutates new_game and switches the turn)
        new_game.make_move(r, c)
        
        # Return the new state wrapper
        return OthelloAIState(new_game)

    def get_current_player(self):
        """Returns the current player (BLACK or WHITE)."""
        return self.game.get_current_player()

    def heuristic_eval(self, maximizing_player):
        """
        A DUMMY HEURISTIC: Returns a fixed value, which will force 
        the alpha-beta search to prioritize the shortest path to a terminal
        node (game over) or simply stop at max depth.
        """
        if self.game.game_over:
            # If terminal, return the actual score difference
            scores = self.game.get_score()
            player_score = scores[maximizing_player]
            opponent = self.WHITE if maximizing_player == self.BLACK else self.BLACK
            opponent_score = scores[opponent]
            
            # Use a very large number for win/loss to ensure the AI prioritizes them
            score_diff = player_score - opponent_score
            return score_diff * 1000  # Multiply to prioritize wins/losses over anything else
        
        # For non-terminal states, return the same dummy value.
        # A positive value encourages any available path until max_depth/game_over.
        return self.DUMMY_HEURISTIC_VALUE

class OthelloAI:
    """
    The main class for the Othello AI.
    """
    def __init__(self, depth: int):
        self.search_engine = AlphaBetaPruning(max_depth=depth)

    def get_ai_move(self, game_state: OthelloGame):
        """
        Takes the current OthelloGame object and returns the best move (r, c).
        """
        # Wrap the current game state for the search engine
        othello_state = OthelloAIState(game_state)
        
        # Run the search
        best_move = self.search_engine.find_best_move(othello_state)
        
        return best_move

# --- Example Usage (Integration with CLI) ---
if __name__ == '__main__':
    # This is a small example of how to use the AI
    print("--- Othello AI Search Test ---")
    
    # Create an Othello game instance
    game = OthelloGame()
    
    # Create an AI instance with a search depth of 3
    # A low depth is used here for a quick test
    ai = OthelloAI(depth=3) 
    
    print(f"Current Player: {game.get_current_player()}")
    print(f"Initial Valid Moves: {list(game.get_valid_moves().keys())}")
    
    # Find the best move
    best_move = ai.get_ai_move(game)
    
    print(f"\nAI (Depth 3) recommends move: {best_move}")
    
    if best_move:
        r, c = best_move
        print(f"Executing move: ({r}, {c})")
        game.make_move(r, c)
        
        print(f"New Player: {game.get_current_player()}")
        print(f"New Score: {game.get_score()}")
    else:
        print("No moves found.")

# --- End of othello_ai.py ---
