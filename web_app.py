from flask import Flask, render_template, request, jsonify
import os
import requests
import logging

app = Flask(__name__)

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://fastapi:8000")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.before_request
def check_api_health():
    """Check API health on first request"""
    if not hasattr(app, 'api_healthy'):
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=3)
            app.api_healthy = response.status_code == 200
        except:
            app.api_healthy = False

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    """Check if FastAPI is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if response.status_code == 200:
            return jsonify(response.json()), 200
        return jsonify({"error": "API not healthy"}), 500
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict", methods=["POST"])
def predict():
    """Call FastAPI prediction endpoint"""
    try:
        data = request.json
        
        # Validate input
        required_fields = ["num_clicks", "days_active", "avg_score", "studied_credits"]
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Call FastAPI
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=data,
            timeout=15
        )
        
        if response.status_code == 200:
            return jsonify(response.json()), 200
        else:
            return jsonify({"error": f"API returned {response.status_code}"}), response.status_code
    
    except requests.exceptions.Timeout:
        return jsonify({"error": "Request timeout"}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({"error": f"Cannot connect to API at {API_BASE_URL}"}), 503
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template("500.html"), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)  # Port 5000 inside container, mapped to 8888 externally
