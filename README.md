# README.md

Coached Learning Aiding Revealment and Identification of Foremost Interest (**CLARIF-i**) is a demo system implementing coachable preference elicitation. A machine learner is fine tuned by its (human) coach with the aim of properly learning and applying their preferences.

## Repository Structure

```plaintext
CLARIF-i
├── Attempt 1
│   └── cpnet_coaching.py
├── Attempt 2
│   ├── control.py
│   ├── display.py
│   ├── engines.py
│   ├── events.py
│   ├── layout.py
│   ├── manager.py
│   ├── run.cmd
│   └── starter.py
├── Attempt 3
│   ├── coachable_search
│   │   ├── core
│   │   │   ├── coach.py
│   │   │   ├── __init__.py
│   │   │   ├── learner.py
│   │   │   └── rule.py
│   │   ├── __init__.py
│   │   └── utils
│   │       ├── convergence.py
│   │       ├── frames.py
│   │       ├── __init__.py
│   │       ├── interface.py
│   │       └── scenarios.py
│   ├── convergence_plot.png
│   ├── README.md
│   ├── setup.py
│   └── test_interface.py
├── Attempt 4
│   ├── core
│   │   ├── coach.py
│   │   └── search.py
│   ├── execute.py
│   ├── gui
│   │   └── app.py
│   ├── logs
│   │   └── app.log
│   ├── requirements.txt
│   ├── utils
│   │   ├── config.py
│   │   └── logger.py
│   └── web
│       ├── app.py
│       └── templates
│           └── index.html
├── CLARIF-i-stable
│   └── api
│       ├── Learner.py
│       ├── Literal.py
│       ├── Rule.py
│       └── State.py
├── README.md
└── related work.txt
```