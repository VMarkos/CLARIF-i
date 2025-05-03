import json
import os
from typing import Dict, List, Any
from ..core.rule import Rule

class ScenarioManager:
    def __init__(self):
        self.scenarios_dir = os.path.join(os.path.dirname(__file__), "..", "scenarios")
        os.makedirs(self.scenarios_dir, exist_ok=True)
        self.current_scenario = None

    def get_scenario_list(self) -> List[str]:
        """Get list of available scenarios."""
        return [f for f in os.listdir(self.scenarios_dir) if f.endswith('.json')]

    def load_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """Load a scenario from file."""
        file_path = os.path.join(self.scenarios_dir, scenario_name)
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Convert rules from dict to Rule objects
        data['learner_rules'] = [Rule.from_dict(rule) for rule in data['learner_rules']]
        data['coach_rules'] = [Rule.from_dict(rule) for rule in data['coach_rules']]
        data['feedback_rules'] = [Rule.from_dict(rule) for rule in data['feedback_rules']]
        
        self.current_scenario = scenario_name
        return data

    def save_scenario(self, scenario_name: str, data: Dict[str, Any]) -> None:
        """Save a scenario to file."""
        # Convert Rule objects to dict
        data_to_save = data.copy()
        data_to_save['learner_rules'] = [rule.to_dict() for rule in data['learner_rules']]
        data_to_save['coach_rules'] = [rule.to_dict() for rule in data['coach_rules']]
        data_to_save['feedback_rules'] = [rule.to_dict() for rule in data['feedback_rules']]
        
        file_path = os.path.join(self.scenarios_dir, scenario_name)
        with open(file_path, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        
        self.current_scenario = scenario_name

    def create_new_scenario(self) -> Dict[str, Any]:
        """Create a new empty scenario."""
        return {
            'state_space': {},
            'start_state': {},
            'goal_state': {},
            'traces': [],
            'learner_rules': [],
            'coach_rules': [],
            'feedback_rules': [],
            'performance_history': []
        }

# Predefined demo scenarios
DEMO_SCENARIOS = {
    'simple_maze': {
        'state_space': {
            'position': {'A', 'B', 'C', 'D', 'E'},
            'has_key': {True, False}
        },
        'start_state': {'position': 'A', 'has_key': False},
        'goal_state': {'position': 'E', 'has_key': True},
        'traces': [
            [({'position': 'A', 'has_key': False}, 'move_right'),
             ({'position': 'B', 'has_key': False}, 'pick_key'),
             ({'position': 'B', 'has_key': True}, 'move_right')]
        ],
        'learner_rules': [
            Rule("Move to key", {"position": "B"}, {"action": "pick_key"}, 1),
            Rule("Move to exit", {"has_key": True}, {"action": "move_right"}, 1)
        ],
        'coach_rules': [
            Rule("Hint about key", {"position": "A"}, {"action": "hint_key"}, 1),
            Rule("Hint about exit", {"position": "D"}, {"action": "hint_exit"}, 1)
        ],
        'feedback_rules': [],
        'performance_history': [0.3, 0.5, 0.7, 0.9]
    },
    'traffic_light': {
        'state_space': {
            'light': {'red', 'yellow', 'green'},
            'position': {'far', 'near', 'at'},
            'speed': {'slow', 'normal', 'fast'}
        },
        'start_state': {'light': 'red', 'position': 'far', 'speed': 'normal'},
        'goal_state': {'light': 'green', 'position': 'at', 'speed': 'slow'},
        'traces': [
            [({'light': 'red', 'position': 'far', 'speed': 'normal'}, 'slow_down'),
             ({'light': 'red', 'position': 'near', 'speed': 'slow'}, 'stop'),
             ({'light': 'green', 'position': 'at', 'speed': 'slow'}, 'go')]
        ],
        'learner_rules': [
            Rule("Slow at red", {"light": "red"}, {"action": "slow_down"}, 1),
            Rule("Stop at light", {"position": "near"}, {"action": "stop"}, 1)
        ],
        'coach_rules': [
            Rule("Warn about red", {"light": "red", "speed": "normal"}, {"action": "warn_slow"}, 1),
            Rule("Remind to stop", {"position": "near", "speed": "fast"}, {"action": "warn_stop"}, 1)
        ],
        'feedback_rules': [],
        'performance_history': [0.4, 0.6, 0.8, 0.95]
    }
} 