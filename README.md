
#**Engineer:** Sai Rohan Kalagiri

## Architecture Overview
This is an end-to-end Machine Learning pipeline built to ingest messy contractor data, predict line-item approval outcomes, and serve predictions via a live inference API with **no handoffs**.

### 1. Data Engineering (The Pipeline)
* Built an object-oriented ingestion pipeline to handle **schema drift** (e.g., mapping scattered columns like `rate` and `qty` into a canonical schema).
* Engineered robust cleaning functions using Regex to strip symbols from messy strings (e.g., converting `"50 sqft"` and `"$120.50"` into clean floats).
* Implemented automated imputation and date-feature extraction.

### 2. Machine Learning 
* Trained a **Gradient-Boosted tabular model** (XGBoost) to predict if a claim will be fully approved or negotiated/rejected.
* Strictly enforced drop logic during training to prevent **target leakage** (hiding the final approved amount from the model).
* Optimized for precision to minimize costly false positives.

### 3. Production Serving (FastAPI)
* Wrapped the XGBoost model and ingestion pipeline into a live **FastAPI** server.
* The API accepts raw, uncleaned JSON payloads, cleans the data on the fly, one-hot encodes categorical variables, and returns a prediction and confidence score in milliseconds.

## How to Run Locally
1. `pip install -r requirements.txt`
2. `uvicorn main:app --host 127.0.0.1 --port 8000`
3. Navigate to `http://127.0.0.1:8000/docs` to test the live API via Swagger UI.
