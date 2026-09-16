import pandas as pd
import numpy as np
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ClaimsDataPipeline:
    def __init__(self):
        # UPDATED: Matching the exact columns in your uploaded CSV
        self.target_schema = [
            'claim_id', 'franchise_id', 'item_category', 
            'date_submitted', 'quantity', 'unit_price', 
            'requested_total', 'carrier_approved_total'
        ]
        
        # UPDATED: Matching the exact schema drift in your uploaded CSV
        self.schema_map = {
            'rate': 'unit_price',
            'qty': 'quantity'
        }

    def _normalize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bulletproof schema normalization."""
        # 1. Guarantee the standard columns exist first
        for standard_col in ['unit_price', 'quantity', 'requested_total']:
            if standard_col not in df.columns:
                df[standard_col] = np.nan
                
        # 2. Map the scattered data into the standard columns
        for weird_name, standard_name in self.schema_map.items():
            if weird_name in df.columns:
                df[standard_name] = df[standard_name].fillna(df[weird_name])
                
        # 3. Only keep the target columns
        available_cols = [col for col in self.target_schema if col in df.columns]
        return df[available_cols]

    def _clean_numeric(self, val) -> float:
        """Strips $, commas, and text units from strings."""
        if pd.isna(val):
            return np.nan
        if isinstance(val, (int, float)):
            return float(val)
        
        cleaned = re.sub(r'[^\d.]', '', str(val))
        try:
            return float(cleaned) if cleaned else np.nan
        except ValueError:
            return np.nan

    def _normalize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forces date chaos into standard YYYY-MM-DD."""
        df['date_submitted'] = pd.to_datetime(df['date_submitted'], errors='coerce')
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates the target variable and ML features."""
        # Calculate approval target based on requested vs approved
        df['is_approved_fully'] = (df['carrier_approved_total'] >= df['requested_total']).astype(int)
        
        # Extract Date Features
        df['submit_month'] = df['date_submitted'].dt.month
        df['submit_day_of_week'] = df['date_submitted'].dt.dayofweek
        
        return df

    def process_training_data(self, file_path: str) -> pd.DataFrame:
        logging.info(f"Loading raw claims data from {file_path}")
        df = pd.read_csv(file_path)
        
        logging.info("Normalizing schema drift...")
        df = self._normalize_schema(df)
        
        logging.info("Cleaning numeric fields...")
        for col in ['quantity', 'unit_price', 'requested_total', 'carrier_approved_total']:
            if col in df.columns:
                df[col] = df[col].apply(self._clean_numeric)
                
        logging.info("Normalizing dates...")
        df = self._normalize_dates(df)
        
        logging.info("Imputing missing data...")
        df['quantity'] = df['quantity'].fillna(1)
        df['unit_price'] = df['unit_price'].fillna(df['unit_price'].median())
        df = df.dropna(subset=['requested_total', 'carrier_approved_total'])
        
        logging.info("Engineering features...")
        # Explicit copy to avoid Pandas warnings
        df = self._engineer_features(df.copy())
        
        logging.info(f"Pipeline complete. Yielded {len(df)} clean records.")
        return df

if __name__ == "__main__":
    pipeline = ClaimsDataPipeline()
    clean_data = pipeline.process_training_data('raw_messy_claims.csv')
    
    clean_data.to_csv('clean_claims_features.csv', index=False)
    
    print("\n--- Clean Data Preview ---")
    print(clean_data[['franchise_id', 'quantity', 'unit_price', 'requested_total', 'is_approved_fully']].head())