from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "model_artifacts"

# ------------------------------------------------------------
# Load saved ML artifacts
# ------------------------------------------------------------
try:
    model = joblib.load(ARTIFACT_DIR / "model.pkl")
    scaler = joblib.load(ARTIFACT_DIR / "scaler.pkl")
    label_encoders = joblib.load(ARTIFACT_DIR / "label_encoders.pkl")
    feature_columns = joblib.load(ARTIFACT_DIR / "feature_columns.pkl")
    preprocessing_metadata = joblib.load(
        ARTIFACT_DIR / "preprocessing_metadata.pkl"
    )

    categorical_cols = preprocessing_metadata["categorical_cols"]
    numerical_cols = preprocessing_metadata["numerical_cols"]

    print("All ML artifacts loaded successfully.")

except Exception as e:
    model = None
    scaler = None
    label_encoders = None
    feature_columns = None
    preprocessing_metadata = None
    categorical_cols = []
    numerical_cols = []

    print("ERROR: Could not load ML artifacts.")
    print(e)


# ------------------------------------------------------------
# Home / health check
# ------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "message": "AI Customer Churn Prediction API is running."
    })


# ------------------------------------------------------------
# Model information
# ------------------------------------------------------------
@app.route("/model-info", methods=["GET"])
def model_info():
    if model is None:
        return jsonify({
            "status": "error",
            "message": "Model artifacts could not be loaded."
        }), 500

    return jsonify({
        "status": "ready",
        "model": type(model).__name__,
        "number_of_features": len(feature_columns),
        "features": feature_columns
    })


# ------------------------------------------------------------
# Exact categorical options from the trained LabelEncoders
# ------------------------------------------------------------
@app.route("/options", methods=["GET"])
def options():
    if label_encoders is None:
        return jsonify({
            "status": "error",
            "message": "Label encoders could not be loaded."
        }), 500

    encoder_options = {}

    for col, encoder in label_encoders.items():
        encoder_options[col] = encoder.classes_.tolist()

    return jsonify({
        "status": "ready",
        "options": encoder_options
    })


# ------------------------------------------------------------
# Prediction endpoint
# ------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({
            "error": "Model is not loaded. Check the model_artifacts folder."
        }), 500

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No JSON data was received."
            }), 400

        required_columns = [
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "tenure",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "MonthlyCharges",
            "TotalCharges"
        ]

        missing = [
            column for column in required_columns
            if column not in data
        ]

        if missing:
            return jsonify({
                "error": "Missing input fields.",
                "missing_fields": missing
            }), 400

        # Build one-row DataFrame in raw input form.
        input_df = pd.DataFrame([{
            column: data[column]
            for column in required_columns
        }])

        # The notebook converted SeniorCitizen from 0/1
        # into No/Yes before label encoding.
        if "SeniorCitizen" in input_df.columns:
            value = str(input_df.loc[0, "SeniorCitizen"]).strip()

            if value.lower() in ["1", "yes", "true"]:
                input_df.loc[0, "SeniorCitizen"] = "Yes"
            elif value.lower() in ["0", "no", "false"]:
                input_df.loc[0, "SeniorCitizen"] = "No"

        # Apply the exact fitted LabelEncoder for each categorical column.
        for col in categorical_cols:

            if col not in label_encoders:
                return jsonify({
                    "error": f"No saved LabelEncoder found for '{col}'."
                }), 500

            encoder = label_encoders[col]

            # Allow the column to hold the integer produced by LabelEncoder.
            input_df[col] = input_df[col].astype(object)

            raw_value = str(input_df.loc[0, col]).strip()

            if raw_value not in encoder.classes_:
                return jsonify({
                    "error": f"Invalid value '{raw_value}' for '{col}'.",
                    "allowed_values": encoder.classes_.tolist()
                }), 400

            encoded_value = encoder.transform([raw_value])[0]
            input_df.loc[0, col] = encoded_value

        # Convert numerical values.
        for col in numerical_cols:
            try:
                input_df[col] = pd.to_numeric(input_df[col])
            except Exception:
                return jsonify({
                    "error": f"Invalid numerical value for '{col}'."
                }), 400

        # Apply the same fitted StandardScaler used during training.
        input_df[numerical_cols] = scaler.transform(
            input_df[numerical_cols]
        )

        # Ensure exact feature order used during model training.
        input_df = input_df[feature_columns]

        # Model prediction.
        prediction = int(model.predict(input_df)[0])

        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(input_df)[0][1])
        else:
            probability = None

        churn = prediction == 1

        if probability is not None:
            if probability >= 0.70:
                risk_level = "High"
            elif probability >= 0.40:
                risk_level = "Medium"
            else:
                risk_level = "Low"

            confidence = round(
                (probability if churn else 1 - probability) * 100,
                2
            )
        else:
            risk_level = "High" if churn else "Low"
            confidence = None

        if churn:
            recommendation = (
                "Customer is at risk of churn. "
                "Consider retention offers, contract incentives "
                "or proactive customer support."
            )
        else:
            recommendation = (
                "Customer is currently predicted to remain. "
                "Continue normal engagement and retention strategies."
            )

        return jsonify({
            "prediction": "Yes" if churn else "No",
            "prediction_label": (
                "Customer Will Churn"
                if churn
                else "Customer Will Stay"
            ),
            "churn_probability": (
                round(probability * 100, 2)
                if probability is not None
                else None
            ),
            "confidence": confidence,
            "risk_level": risk_level,
            "recommendation": recommendation
        })

    except Exception as e:
        return jsonify({
            "error": "Prediction failed.",
            "details": str(e)
        }), 500


# ------------------------------------------------------------
# Run locally
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
