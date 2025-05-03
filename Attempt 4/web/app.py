"""
Web interface for the Coach-Driven Search application.
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sys
import os

# Add parent directory to path to import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.search import SearchEngine
from core.coach import Coach
from utils.config import load_config
from utils.logger import setup_logger

app = Flask(__name__)
CORS(app)

# Initialize components
config = load_config()
search_engine = SearchEngine(config)
coach = Coach(config)
logger = setup_logger()

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    """Handle search requests."""
    data = request.get_json()
    query = data.get('query', '')
    
    # Get coach's analysis
    analysis = coach.analyze_query(query)
    
    # Get search results
    results = search_engine.search(query)
    
    # Get improvement suggestions
    suggestions = coach.suggest_improvements(query)
    
    return jsonify({
        'analysis': analysis,
        'results': results,
        'suggestions': suggestions
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000) 