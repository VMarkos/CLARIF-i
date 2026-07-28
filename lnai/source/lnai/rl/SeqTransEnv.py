# rl/SeqTransEnv.py
#
# Sequence transformation environment

import numpy as np
from gymnasium import Env, spaces
from sb3_contrib import MaskablePPO
from lnai.rl.SortingCoach import SortingCoach
from lnai.rl.SortingEnv import SortingEnv
from itertools import combinations

class SeqTransEnv(Env):
    def __init__(self, max_n: int, sorter: MaskablePPO, coach: SortingCoach, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_n = max(2, max_n)
        self.sorter = sorter
        self.coach = coach
        # Number of steps after which model execution should stop
        self._cutoff = 100
        self._max_swaps = max_n * (max_n - 1) // 2 # Number of possible swaps for the maximum given value of n
        self._max_path_len = max_n * (max_n - 1) # Double the theoretical maximum for Bubble-sort
        self._swap_pairs = list(combinations(range(self.max_n), 2))
        self._edit_masker = PathEditActionMasker(max_n)

        # Observation and action spaces
        self.observation_space = spaces.Dict({
            'local_context': spaces.Box(
                low=-1, high=max_n - 1, shape=(3, self.max_n), dtype=np.int32,
            ),
            'source_path': spaces.Box(
                low=-1, high=max_n - 1, shape=(self._max_path_len, self.max_n), dtype=np.int32,
            ),
            'target_path': spaces.Box(
                low=-1, high=max_n - 1, shape=(self._max_path_len, self.max_n), dtype=np.int32,
            ),
            'edit_status': spaces.Box(
                low=0, high=self._max_path_len, shape=(3,), dtype=np.int32,
            ),
        })
        self.action_space = spaces.Discrete(2 + 1 + 2 * self._max_swaps)

        self.reset()


    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed, options=options)
        # Initialise internal clock
        self._ticks = 0
        self._cursor = 1

        # Create random permutation and sort it
        n = np.random.choice(np.arange(2, self.max_n + 1))
        state = np.arange(self.max_n)
        state[:n] = np.random.permutation(n)
        self._current_trace = self._run_model(state.copy())

        # Get coach advice
        self.coach.update_state(state)
        self._target_trace = self.coach.get_trace()

        self._prev_distance = self._compute_trace_distance(self._current_trace, self._target_trace)

        return self._get_obs(), dict()
        


    def render(self) -> None:
        print(f"Tick: {self._ticks} | Cursor: {self._cursor} | Trace Len: {len(self._current_trace)}")


    def step(self, action: int) -> tuple:
        self._ticks += 1
        terminated = False
        truncated = False
        if action == 0:
            self._cursor = max(1, self._cursor - 1)
        elif action == 1:
            self._cursor = min(len(self._current_trace) - 2, self._cursor + 1)
        elif action == 2:
            if 0 < self._cursor < len(self._current_trace) - 1 and len(self._current_trace) > 2:
                del self._current_trace[self._cursor]
                self._cursor = min(self._cursor, len(self._current_trace) - 2)
        elif action < 3 + self._max_swaps:
            swap_i = action - 3
            i, j = self._swap_pairs[swap_i]
            curr_state = self._current_trace[self._cursor].copy()
            curr_state[i], curr_state[j] = curr_state[j], curr_state[i]
            self._current_trace.insert(self._cursor + 1, curr_state)
            self._cursor += 1
        else:
            swap_i = action - 3 - self._max_swaps
            i, j = self._swap_pairs[swap_i]
            curr_state = self._current_trace[self._cursor - 1].copy()
            curr_state[i], curr_state[j] = curr_state[j], curr_state[i]
            self._current_trace[self._cursor] = curr_state

        # self._cursor = int(np.clip(self._cursor, 1, max(1, len(self._current_trace) - 2)))
        current_distance = self._compute_trace_distance(self._current_trace, self._target_trace)
        delta = self._prev_distance - current_distance
        reward = -0.005

        # 1. Give small positive feedback for reducing distance, but don't heavily penalize temporary spikes
        if delta > 0:
            reward += delta * 2.0  # Positive progress
        elif delta < 0:
            reward -= 0.5  # Soft penalty for increasing distance (not hard penalty!)

        # 2. Add an explicit edit incentive (reward structural edits that stay valid)
        if action >= 2 and delta == 0.0:  # DELETE, INSERT, or SUBSTITUTE
            reward -= 0.02

        if action < 2:
            reward += 0.005

        self._prev_distance = current_distance


        if current_distance == 0.0:
            reward += 10.0
            terminated = True

        if self._ticks >= self._cutoff or len(self._current_trace) >= self._max_path_len:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, dict()


    def action_masks(self) -> np.ndarray:
        mask = np.zeros(2 + 1 + 2 * self._max_swaps, dtype=bool)
        
        edit_masker = self._edit_masker.compute_action_mask(self._current_trace, self._cursor)

        # Move Left
        mask[0] = self._cursor > 1

        # Move Right
        mask[1] = self._cursor < len(self._current_trace) - 2

        # Delete
        mask[2] = edit_masker["can_delete"]
        
        # Insert (indices 1 to num_swaps)
        mask[3 : 3 + self._max_swaps] = edit_masker["valid_insertions"]
        
        # Substitute (indices 1 + num_swaps to 2*num_swaps)
        mask[3 + self._max_swaps :] = edit_masker["valid_substitutions"]
        
        return mask


    def _get_obs(self) -> dict:
        """Pads traces and constructs the fixed-shape observation dictionary."""
        # 1. Local Context: [S_{i-1}, S_i, S_{i+1}]
        local_context = np.full((3, self.max_n), -1, dtype=np.int32)
        if len(self._current_trace) > 0:
            local_context[0] = self._current_trace[self._cursor - 1]
            local_context[1] = self._current_trace[self._cursor]
            if self._cursor + 1 < len(self._current_trace):
                local_context[2] = self._current_trace[self._cursor + 1]

        # 2. Source Path (Padded)
        source_path = np.full((self._max_path_len, self.max_n), -1, dtype=np.int32)
        src_len = min(len(self._current_trace), self._max_path_len)
        if src_len > 0:
            source_path[:src_len] = np.array(self._current_trace[:src_len])

        # 3. Target Path (Padded)
        target_path = np.full((self._max_path_len, self.max_n), -1, dtype=np.int32)
        tgt_len = min(len(self._target_trace), self._max_path_len)
        if tgt_len > 0:
            target_path[:tgt_len] = np.array(self._target_trace[:tgt_len])

        # 4. Edit Status
        edit_status = np.array([self._cursor, src_len, tgt_len], dtype=np.int32)

        return {
            'local_context': local_context,
            'source_path': source_path,
            'target_path': target_path,
            'edit_status': edit_status,
        }


    def _compute_trace_distance(self, path_a: list[np.ndarray], path_b: list[np.ndarray]) -> float:
        """Computes sequence Levenshtein distance between two state trajectories."""
        m, n = len(path_a), len(path_b)
        dp = np.zeros((m + 1, n + 1), dtype=int)
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if np.array_equal(path_a[i - 1], path_b[j - 1]):
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j],      # Deletion
                                       dp[i][j - 1],      # Insertion
                                       dp[i - 1][j - 1])  # Substitution
        return float(dp[m][n])


    def _run_model(self, start_state: np.ndarray) -> list[np.ndarray]:
        env = SortingEnv(len(start_state))
        obs, _ = env.reset(start_state)
        trace = [obs.copy()]
        for i in range(self._cutoff):
            action, _ = self.sorter.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            trace.append(obs.copy())
            if terminated or truncated:
                return trace
        return trace




class PathEditActionMasker:
    def __init__(self, n: int):
        self.n = n
        # Pre-generate all n*(n-1)/2 swap pairs (i, j)
        self.swap_pairs = list(combinations(range(n), 2))
        self.num_swaps = len(self.swap_pairs)

    def _get_one_swap_neighbors(self, state: np.ndarray) -> list[np.ndarray]:
        """Generates all n*(n-1)/2 valid 1-swap neighbor states from current state."""
        neighbors = []
        for i, j in self.swap_pairs:
            neighbor = state.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbors.append(neighbor)
        return neighbors

    def compute_action_mask(
        self, path: list[np.ndarray], cursor_idx: int
    ) -> dict[str, bool | np.ndarray]:
        """
        Computes boolean masks for Delete, Insert, and Substitute at cursor_idx.
        
        Args:
            path: Current state sequence [S_0, S_1, ..., S_m]
            cursor_idx: Index i where the editing action is applied (1 <= i < len(path)-1)
            
        Returns:
            Dict containing:
              - 'can_delete': bool
              - 'valid_insertions': boolean array of shape (num_swaps,)
              - 'valid_substitutions': boolean array of shape (num_swaps,)
        """
        m = len(path)
        if cursor_idx <= 0 or cursor_idx >= len(path) - 1:
            return {
                "can_delete": False,
                "valid_insertions": np.zeros(self.num_swaps, dtype=bool),
                "valid_substitutions": np.zeros(self.num_swaps, dtype=bool),
            }

        prev_state = path[cursor_idx - 1]
        curr_state = path[cursor_idx]
        next_state = path[cursor_idx + 1]

        # 1. DELETION MASK
        # Deleting curr_state is valid iff prev_state and next_state are 1 swap apart
        can_delete = np.count_nonzero(prev_state != next_state) == 2

        # Generate candidates reachable in 1 swap from prev_state
        candidates_from_prev = self._get_one_swap_neighbors(prev_state)

        # 2. INSERTION MASK (Inserting S_new between curr_state and next_state)
        # S_new must be 1 swap from curr_state AND 1 swap from next_state
        candidates_from_curr = self._get_one_swap_neighbors(curr_state)
        valid_insertions = np.zeros(self.num_swaps, dtype=bool)

        for act_idx, cand in enumerate(candidates_from_curr):
            # Check distance to next_state
            if np.count_nonzero(cand != next_state) == 2 and not np.array_equal(cand, curr_state) and not np.array_equal(cand, next_state):
                valid_insertions[act_idx] = True

        # 3. SUBSTITUTION MASK (Replacing curr_state with S_new)
        # S_new must be 1 swap from prev_state AND 1 swap from next_state
        valid_substitutions = np.zeros(self.num_swaps, dtype=bool)

        for act_idx, cand in enumerate(candidates_from_prev):
            # Check distance to next_state
            if np.count_nonzero(cand != next_state) == 2 and not np.array_equal(cand, curr_state):
                valid_substitutions[act_idx] = True
                

        return {
            "can_delete": can_delete,
            "valid_insertions": valid_insertions,
            "valid_substitutions": valid_substitutions,
        }
