import joblib
import pandas as pd

MODEL_PATH = "models/rf_crop_model.pkl"

crop_dict = {
    0: "rice",
    1: "wheat",
    2: "maize",
    3: "cotton",
    4: "sugarcane",
    5: "tobacco",
    6: "millet",
    7: "barley",
    8: "peas",
    9: "groundnut",
    10: "soybean",
    11: "chili",
    12: "onion",
    13: "garlic",
    14: "potato",
    15: "tomato",
    16: "cabbage",
    17: "cauliflower",
    18: "carrot",
    19: "beetroot",
    20: "radish",
    21: "spinach"
}

def predict_crop(features: dict):
    """
    Predict the best crop based on soil & weather features.
    features: dict with keys 'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'
    Returns: predicted crop label and confidence
    """
    model = joblib.load(MODEL_PATH)
    
    # Convert input dict to DataFrame with exact column names used during training
    X = pd.DataFrame([features], columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
    
    pred = model.predict(X)[0]
    proba = model.predict_proba(X).max()
    
    return crop_dict[pred], proba
