from coachable_search.core.rule import Rule
from coachable_search.core.learner import Learner
from coachable_search.core.coach import Coach
from coachable_search.utils.interface import CoachingInterface

def main():
    # Initialize interface
    interface = CoachingInterface()
    
    # Run the interface
    interface.run()

if __name__ == "__main__":
    main() 