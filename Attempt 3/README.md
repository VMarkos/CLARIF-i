# Coachable Search

A framework for coach-learner interaction and learning, where a coach guides a learner to develop a hypothesis that aligns with the coach's target rules.

## Overview

The framework implements a learning process where:
1. A coach maintains a set of target rules representing desired behavior
2. A learner maintains a hypothesis (set of rules) that evolves over time
3. The learner makes inferences based on their current hypothesis
4. The coach observes the inferences and provides feedback
5. The learner updates their hypothesis based on the feedback
6. The process continues until the learner's hypothesis converges to the coach's target rules

## Installation

```bash
pip install -e .
```

## Usage

The framework can be used to implement various coach-learner scenarios. Here's a simple example:

```python
from coachable_search import Rule, Coach, Learner, ConvergenceTracker, run_coaching_session

# Define target rules
target_rules = [
    Rule(
        condition={"shape": "circle"},
        preference={"color": "red"},
        priority=2,
        explanation="Circles should be red"
    ),
    # ... more rules ...
]

# Create coach and learner
coach = Coach(target_rules)
learner = Learner([])
tracker = ConvergenceTracker()

# Define variable domains
variable_domains = {
    "shape": {"circle", "square", "triangle"},
    "color": {"red", "blue", "green"},
    "size": {"small", "large"}
}

# Run coaching session
converged, iterations = run_coaching_session(
    coach, learner, tracker, variable_domains
)
```

## Structure

The package is organized as follows:

- `coachable_search/`
  - `core/`
    - `coach.py`: Coach implementation
    - `learner.py`: Learner implementation
    - `rule.py`: Rule data structure
  - `utils/`
    - `convergence.py`: Convergence tracking
    - `session.py`: Session management

## Demo

Run the demonstration:

```bash
python demo_coaching.py
```

This will:
1. Create a coach with target rules about shapes, colors, and sizes
2. Start with a learner that has no initial rules
3. Run a coaching session
4. Track and visualize the convergence process
5. Save a plot showing how the learner's success rate improves over time

## License

MIT License 