import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory to path

# mcp_server/server.py
from flask import Flask, request, jsonify
from tools import clean_data, generate_summary, run_prediction

app = Flask(__name__)

@app.route('/clean_data', methods=['POST'])
def clean_data_endpoint():
    data = request.json.get('data')
    cleaned_data = clean_data(data)  # Assuming clean_data is a tool function
    return jsonify({"cleaned_data": cleaned_data})

@app.route('/generate_summary', methods=['POST'])
def generate_summary_endpoint():
    data = request.json.get('data')
    summary = generate_summary(data)  # Assuming generate_summary is a tool function
    return jsonify({"summary": summary})

@app.route('/run_prediction', methods=['POST'])
def run_prediction_endpoint():
    data = request.json.get('data')
    target = request.json.get('target_column')
    prediction_results = run_prediction(data, target)  # Assuming run_prediction is a tool function
    return jsonify({"prediction_results": prediction_results})

if __name__ == "__main__":
    app.run(debug=True)
