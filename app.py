import os
from flask import Flask, request, jsonify, render_template
from joblib import load
import pandas as pd

MODEL_PATH = os.environ.get("CAR_MODEL_PATH", "./models/car_price_pipeline.pkl")

app = Flask(__name__)
model = load(MODEL_PATH)

EXPECTED_COLS = [
    "name","fuel","seller_type","transmission","owner",
    "year","km_driven","mileage","engine","max_power","seats"
]

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", expected_cols=EXPECTED_COLS)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form if request.form else request.json
    if data is None:
        return jsonify({"error": "No input provided."}), 400

    row = {col: data.get(col, None) for col in EXPECTED_COLS}
    df = pd.DataFrame([row])

    for col in ["year","km_driven","mileage","engine","max_power","seats"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    try:
        pred = model.predict(df)[0]
        return jsonify({"predicted_price": float(pred)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)