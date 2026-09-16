import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

def train_approval_model():
    print("Loading clean features...")
    df = pd.read_csv('clean_claims_features.csv')

    # 1. Drop Target Leakage and Identifiers
    # We drop the answer (carrier_approved_total) and things that don't help prediction
    features_to_drop = ['claim_id', 'date_submitted', 'carrier_approved_total', 'is_approved_fully']
    
    X = df.drop(columns=[col for col in features_to_drop if col in df.columns])
    y = df['is_approved_fully']

    # 2. Handle Categorical Data (One-Hot Encoding)
    # UPDATED: We changed these to match your actual dataset columns
    print("Encoding categorical variables...")
    X = pd.get_dummies(X, columns=['franchise_id', 'item_category'])

    # Save the exact feature columns so our FastAPI endpoint knows the schema later
    joblib.dump(list(X.columns), 'model_features.pkl')

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    # 4. Evaluation 
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)
    
    # We focus heavily on Precision and Recall here.
    print(classification_report(y_test, y_pred))

    # 5. Save the Model for Production
    model.save_model('claims_xgboost_model.json')
    print("\nModel saved to 'claims_xgboost_model.json'. Ready for production serving!")

if __name__ == "__main__":
    train_approval_model()