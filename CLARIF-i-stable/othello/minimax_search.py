# minimax_search.py

class AbstractState:
    """
    An abstract base class that any game state object must implement
    to be used with the AlphaBetaPruning class.
    """
    def is_terminal(self) -> bool:
        """Returns True if the game is over."""
        raise NotImplementedError

    def get_moves(self):
        """Returns a list of possible moves/actions from the current state."""
        raise NotImplementedError

    def make_move(self, move):
        """Returns a *new* State object resulting from the move."""
        raise NotImplementedError

    def get_current_player(self):
        """Returns the player whose turn it is (e.g., BLACK or WHITE)."""
        raise NotImplementedError

    def heuristic_eval(self, maximizing_player):
        """
        Evaluates the state. A higher value is better for the maximizing_player.
        """
        raise NotImplementedError

class AlphaBetaPruning:
    """
    A generic implementation of the Minimax algorithm with Alpha-Beta Pruning.
    It operates on AbstractState objects.
    """
    def __init__(self, max_depth: int):
        self.max_depth = max_depth
        self.maximizing_player = None

    def find_best_move(self, state: AbstractState):
        """
        Initiates the search from the current state and returns the best move.
        """
        if state.is_terminal():
            return None # No moves possible

        self.maximizing_player = state.get_current_player()
        
        best_value = float('-inf')
        best_move = None

        for move in state.get_moves():
            next_state = state.make_move(move)
            # The search starts on the opponent's turn (Min node)
            value = self._minimax(next_state, 
                                  depth=1, 
                                  alpha=float('-inf'), 
                                  beta=float('inf'))
            
            if value > best_value:
                best_value = value
                best_move = move
                
        return best_move

    def _minimax(self, state: AbstractState, depth: int, alpha: float, beta: float):
        """
        The recursive helper function for Minimax with Alpha-Beta Pruning.
        """
        # 1. Check for terminal states or depth limit
        if depth == self.max_depth or state.is_terminal():
            # Return the heuristic evaluation
            return state.heuristic_eval(self.maximizing_player)

        # 2. Determine if it's a Max or Min node
        if state.get_current_player() == self.maximizing_player:
            # MAX Node
            value = float('-inf')
            for move in state.get_moves():
                next_state = state.make_move(move)
                value = max(value, self._minimax(next_state, depth + 1, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break # Beta cut-off
            return value
        else:
            # MIN Node
            value = float('inf')
            for move in state.get_moves():
                next_state = state.make_move(move)
                value = min(value, self._minimax(next_state, depth + 1, alpha, beta))
                beta = min(beta, value)
                if alpha >= beta:
                    break # Alpha cut-off
            return value

# --- End of minimax_search.py ---
