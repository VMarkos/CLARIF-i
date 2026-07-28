import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CurriculumPathSearchEnv(gym.Env):
    def __init__(self, max_n=6, current_n=3, max_path_len=20):
        super().__init__()
        self.max_n = max_n
        self.max_path_len = max_path_len
        self.current_n = min(current_n, max_n)

        self.max_swaps = max_n * (max_n - 1) // 2
        self.action_space = spaces.Discrete(self.max_swaps + 1)
        self.BACKTRACK_ACTION = self.max_swaps
        self.all_swaps = [
            (i, j)
            for j in range(1, max_n)
                for i in range(j)
        ]
        self.current_swaps = self.current_n * (self.current_n - 1) // 2

        self.observation_space = spaces.Dict({
            "current_array": spaces.Box(low=-1, high=max_n-1, shape=(max_n,), dtype=np.int32),
            "active_mask": spaces.Box(low=0, high=1, shape=(max_n,), dtype=np.int32),
            "action_mask": spaces.Box(low=0, high=1, shape=(self.max_swaps + 1,), dtype=np.int8)
        })

        self.reset()

    def set_curriculum_level(self, new_n: int):
        """Update curriculum level and immediately reset state to match the new dimension."""
        new_n = min(new_n, self.max_n)
        if new_n != self.current_n:
            self.current_n = new_n
            self.current_swaps = self.current_n * (self.current_n - 1) // 2
            self.reset()  # Resets path_stack, target_state, and exploration memory to match new_n


    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.max_swaps + 1, dtype=bool)

        # 1. Allow valid swaps for current n
        active_swaps_count = self.current_n - 1
        mask[:active_swaps_count] = True

        # 2. Mask out actions already tried from current depth
        current_depth = len(self.path_stack) - 1
        if current_depth in self.explored_actions_at_depth:
            for tried_action in self.explored_actions_at_depth[current_depth]:
                if tried_action < self.max_swaps:
                    mask[tried_action] = False

        # 3. Allow BACKTRACK ONLY if we are deeper than root
        can_backtrack = len(self.path_stack) > 1
        mask[self.BACKTRACK_ACTION] = can_backtrack

        # 4. Fallback: If ALL actions masked
        if not np.any(mask):
            if can_backtrack:
                mask[self.BACKTRACK_ACTION] = True
            else:
                # At root and out of moves: unmask any valid swap to prevent zero-mask crash
                mask[:active_swaps_count] = True

        return mask


    def _get_inversion_count(self, arr):
        inv = 0
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                if arr[i] > arr[j]:
                    inv += 1
        return inv

    def _get_obs(self):
        curr_arr = self.path_stack[-1]
        padded_array = np.full(self.max_n, -1, dtype=np.int32)
        padded_array[:self.current_n] = curr_arr

        active_mask = np.zeros(self.max_n, dtype=np.int32)
        active_mask[:self.current_n] = 1

        return {
            "current_array": padded_array,
            "active_mask": active_mask,
            "action_mask": self.action_masks().astype(np.int8)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        initial_arr = np.random.permutation(self.current_n)
        self.target_state = np.arange(self.current_n)
        
        self.path_stack = [initial_arr]
        
        # Track explored action indices per depth level in the search stack
        # Depth 0 -> set of actions tried from root, Depth 1 -> actions tried from step 1, etc.
        self.explored_actions_at_depth = {0: set()}
        self.current_inversions = self._get_inversion_count(initial_arr)
        self.max_path_len = self.current_n * (self.current_n - 1) // 2 * 3

        return self._get_obs(), {}

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)
        mask = self.action_masks()
        if not mask[action]:
            return self._get_obs(), -1.0, False, False, {"invalid_action": True}

        terminated = False
        truncated = False
        reward = 0.0

        current_depth = len(self.path_stack) - 1
        # CASE 1: BACKTRACK
        if action == self.BACKTRACK_ACTION:
            # Safety Check: Never pop the root element (depth 0)
            if len(self.path_stack) > 1:
                if current_depth in self.explored_actions_at_depth:
                    del self.explored_actions_at_depth[current_depth]

                self.path_stack.pop()
                new_state = self.path_stack[-1]
                
                new_inv = self._get_inversion_count(new_state)
                reward = -0.1
                self.current_inversions = new_inv
            else:
                # Tried to backtrack at root (dead end) -> Terminate or truncate episode
                reward = -1.0
                truncated = True

        # CASE 2: FORWARD SWAP
        else:
            # Mark action as explored at this depth
            if current_depth not in self.explored_actions_at_depth:
                self.explored_actions_at_depth[current_depth] = set()
            self.explored_actions_at_depth[current_depth].add(action)

            i, j = self.all_swaps[action]
            curr = self.path_stack[-1].copy()
            curr[i], curr[j] = curr[j], curr[i]

            self.path_stack.append(curr)
            new_depth = len(self.path_stack) - 1
            self.explored_actions_at_depth[new_depth] = set()

            new_inv = self._get_inversion_count(curr)
            # Continuous Potential Reward: Reward progress toward target
            reward = 1.0 * (self.current_inversions - new_inv) - 0.05
            self.current_inversions = new_inv

            if np.array_equal(curr, self.target_state):
                reward += 10.0
                terminated = True

        if len(self.path_stack) >= self.max_path_len:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}


    def render(self) -> None:
        print(
            ' -> '.join(map(str, self.path_stack))
        )
