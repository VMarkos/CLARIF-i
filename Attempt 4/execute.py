#!/usr/bin/env python3
"""
Main entry point for the Coach-Driven Search application.
"""

from core.search import SearchEngine
from core.coach import Coach
from utils.config import load_config
from utils.logger import setup_logger

def main():
    """Main execution function."""
    # Setup logging
    logger = setup_logger()
    logger.info("Starting Coach-Driven Search application")
    
    # Load configuration
    config = load_config()
    
    # Initialize components
    search_engine = SearchEngine(config)
    coach = Coach(config)
    
    # Interactive loop
    while True:
        query = input("\nEnter your search query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
            
        # Get coach's analysis
        analysis = coach.analyze_query(query)
        print(f"\nCoach's Analysis: {analysis}")
        
        # Get search results
        results = search_engine.search(query)
        print(f"\nSearch Results: {results}")
        
        # Get improvement suggestions
        suggestions = coach.suggest_improvements(query)
        print(f"\nSuggested Improvements: {suggestions}")
    
    logger.info("Application completed successfully")

if __name__ == "__main__":
    main() 