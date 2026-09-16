from fastapi import FastAPI
import pandas as pd
import xgboost as xgb
import joblib
from pipeline import ClaimsDataPipeline

app = FastAPI(title="Cozmo Claims Pricing Engine", description="Live Inference API")

# Load our saved model and features when the server starts
print("Loading model and pipeline into memory...")
model = xgb.XGBClassifier()
model.load_model('claims_xgboost_model.json')
model_features = joblib.load('model_features.pkl')
pipeline = ClaimsDataPipeline()

@app.post("/predict")
def predict_approval(live_claim: dict):
    """Accepts a raw, messy claim JSON and returns a prediction."""
    
    # 1. Convert the incoming JSON payload into a Pandas DataFrame
    df = pd.DataFrame([live_claim])
    
    # 2. Run it through the exact same ingestion pipeline we built for training
    df = pipeline._normalize_schema(df)
    
    for col in ['quantity', 'unit_price', 'requested_total']:
        if col in df.columns:
            df[col] = df[col].apply(pipeline._clean_numeric)
            
    df = pipeline._normalize_dates(df)
    
    # Impute missing values for this specific live claim
    df['quantity'] = df['quantity'].fillna(1)
    if 'unit_price' in df.columns and pd.isna(df['unit_price'].iloc[0]):
        df['unit_price'] = 0 
        
    # Extract date features (the model needs these!)
    df['submit_month'] = df['date_submitted'].dt.month
    df['submit_day_of_week'] = df['date_submitted'].dt.dayofweek
    
    # 3. One-Hot Encode categories
    df_encoded = pd.get_dummies(df, columns=['franchise_id', 'item_category'] if 'franchise_id' in df.columns else [])
    
    # 4. Align with training schema 
    # (If the live claim is missing a column the model expects, fill it with 0)
    for col in model_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    # Reorder columns to perfectly match what XGBoost expects
    X_live = df_encoded[model_features]
    
    # 5. Make the Prediction!
    prediction = model.predict(X_live)[0]
    probability = model.predict_proba(X_live)[0][1]
    
    return {
        "claim_id": live_claim.get("claim_id", "UNKNOWN"),
        "status": "APPROVED" if prediction == 1 else "NEGOTIATED/REJECTED",
        "confidence_score": round(float(probability), 4)
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
    