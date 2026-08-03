import os
import sys
from flask import Flask, render_template, jsonify, request

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.inference import PharmaSentinelPredictor

app = Flask(__name__)
predictor = PharmaSentinelPredictor()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/drugs", methods=["GET"])
def get_drugs():
    drugs = predictor.get_all_drugs()
    return jsonify({"success": True, "count": len(drugs), "drugs": drugs})

@app.route("/api/predict", methods=["POST", "GET"])
def predict():
    data = request.get_json(force=True, silent=True) or request.form or request.args or {}
    
    # Handle array of drugs or individual drug parameters
    drugs = data.get("drugs")
    if not drugs:
        d1 = data.get("drug1")
        d2 = data.get("drug2")
        d3 = data.get("drug3")
        d4 = data.get("drug4")
        drugs = [d for d in [d1, d2, d3, d4] if d]

    if not drugs or len(drugs) == 0:
        return jsonify({"success": False, "error": "At least 1 drug must be provided."}), 400

    try:
        res = predictor.predict_regimen(drugs)
        return jsonify({"success": True, "result": res})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    print("\n==========================================================")
    print("  PharmaSentinel Multi-Drug Web Server Running at:       ")
    print("  http://127.0.0.1:5000/                                  ")
    print("==========================================================\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
