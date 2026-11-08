import os

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/machine_failure_model.pkl")
THRESHOLD = 0.5  # probability at or above this is reported as "failure likely"

# HTML form field name -> model feature name
FORM_TO_FEATURE = {
    "type": "Type",
    "air_temp": "Air temperature [K]",
    "process_temp": "Process temperature [K]",
    "rpm": "Rotational speed [rpm]",
    "torque": "Torque [Nm]",
    "tool_wear": "Tool wear [min]",
}

# IMPORTANT: use the same encoding you used in training.
# Ordinal L < M < H is shown here. If you used sklearn's LabelEncoder,
# the mapping is alphabetical instead: {"H": 0, "L": 1, "M": 2}.
# If your model is a Pipeline with its own encoder, delete the mapping and
# pass the raw letter ("L", "M", "H") instead.
TYPE_MAP = {"L": 0, "M": 1, "H": 2}

# Accepted ranges for the numeric inputs (same as the sliders in index.html).
RANGES = {
    "air_temp": (295.0, 305.0),
    "process_temp": (305.0, 315.0),
    "rpm": (1100.0, 2900.0),
    "torque": (3.0, 80.0),
    "tool_wear": (0.0, 260.0),
}

model = joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class InputError(ValueError):
    """Raised when the submitted values are missing or out of range."""


def parse_inputs(data):
    """Validate raw form/JSON data and return a dict keyed by form field name."""
    cleaned = {}

    machine_type = str(data.get("type", "")).strip().upper()
    if machine_type not in TYPE_MAP:
        raise InputError("Choose a product quality variant: L, M or H.")
    cleaned["type"] = machine_type

    for field, (low, high) in RANGES.items():
        raw = data.get(field)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise InputError(f"Enter a number for {FORM_TO_FEATURE[field]}.")
        if not low <= value <= high:
            raise InputError(
                f"{FORM_TO_FEATURE[field]} must be between {low:g} and {high:g}."
            )
        cleaned[field] = value

    return cleaned


def run_prediction(cleaned):
    """Return (failure: bool, probability: float) for validated inputs."""
    X = np.array(
        [[
            TYPE_MAP[cleaned["type"]],
            cleaned["air_temp"],
            cleaned["process_temp"],
            cleaned["rpm"],
            cleaned["torque"],
            cleaned["tool_wear"],
        ]],
        dtype=np.float64,
    )

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(X)[0][1])
    else:
        # Model has no probabilities: fall back to the hard class (0 or 1).
        probability = float(model.predict(X)[0])

    return probability >= THRESHOLD, probability


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form.to_dict()
    try:
        cleaned = parse_inputs(form)
        failure, probability = run_prediction(cleaned)
    except InputError as exc:
        return render_template("index.html", form=form, error=str(exc)), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return (
            render_template(
                "index.html",
                form=form,
                error="Something went wrong while making the prediction. Try again.",
            ),
            500,
        )

    result = {"failure": failure, "probability": probability}
    return render_template("index.html", form=form, result=result)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON endpoint, e.g.
    curl -X POST localhost:5000/api/predict -H "Content-Type: application/json" \
         -d '{"type":"M","air_temp":298.1,"process_temp":308.6,"rpm":1551,"torque":42.8,"tool_wear":0}'
    """
    data = request.get_json(silent=True) or {}
    try:
        cleaned = parse_inputs(data)
        failure, probability = run_prediction(cleaned)
    except InputError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed."}), 500

    return jsonify({"failure": failure, "probability": round(probability, 4)})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0",port=8080)