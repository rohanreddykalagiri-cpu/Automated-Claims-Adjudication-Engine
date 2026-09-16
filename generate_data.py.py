import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# 1. Define the Domain (Water Damage Claims)
categories = ['Water Extraction', 'Drywall Repair', 'Dehumidifier Rental', 'Antimicrobial Treatment', 'Baseboard Replacement']
franchises = ['Franchise_TX_01', 'Franchise_FL_04', 'Franchise_NY_12']

def generate_base_data(num_records=10000):
    data = []
    start_date = datetime(2025, 1, 1)
    
    for i in range(num_records):
        franchise = random.choice(franchises)
        category = random.choice(categories)
        
        # Base pricing logic
        unit_price = round(random.uniform(15.0, 150.0), 2)
        quantity = random.randint(1, 50)
        requested_total = round(unit_price * quantity, 2)
        
        # Simulate the "Negotiated Outcome" (Label)
        # Some items are approved in full, some are slashed by the carrier
        is_approved_in_full = random.choices([True, False], weights=[0.6, 0.4])[0]
        if is_approved_in_full:
            carrier_approved_total = requested_total
            outcome = 1 # Approved
        else:
            carrier_approved_total = round(requested_total * random.uniform(0.4, 0.9), 2)
            outcome = 0 # Negotiated/Rejected
            
        data.append({
            'claim_id': f"CLM-{random.randint(10000, 99999)}",
            'date_submitted': start_date + timedelta(days=random.randint(0, 365)),
            'franchise_id': franchise,
            'item_category': category,
            'unit_price': unit_price,
            'quantity': quantity,
            'requested_total': requested_total,
            'carrier_approved_total': carrier_approved_total,
            'target_outcome': outcome 
        })
    return pd.DataFrame(data)

def inject_messiness(df):
    """Simulates the 'raw sprawl' and schema drift mentioned in the JD."""
    messy_rows = []
    
    for _, row in df.iterrows():
        messy_row = row.copy()
        franchise = row['franchise_id']
        
        # Franchise NY formats dates differently and adds $ signs
        if franchise == 'Franchise_NY_12':
            messy_row['date_submitted'] = row['date_submitted'].strftime('%m/%d/%Y')
            messy_row['unit_price'] = f"${row['unit_price']}"
            # Schema drift: NY calls it 'qty' instead of 'quantity'
            messy_row['qty'] = messy_row.pop('quantity') 
            
        # Franchise FL randomly drops the requested_total (system glitch)
        elif franchise == 'Franchise_FL_04':
            messy_row['date_submitted'] = row['date_submitted'].strftime('%Y-%m-%d')
            if random.random() < 0.1:
                messy_row['requested_total'] = np.nan
            # Schema drift: FL uses 'rate' instead of 'unit_price'
            messy_row['rate'] = messy_row.pop('unit_price')
            
        # Franchise TX leaves string artifacts in categories
        else:
            messy_row['date_submitted'] = row['date_submitted'].strftime('%B %d, %Y')
            if random.random() < 0.2:
                messy_row['item_category'] = f"{row['item_category']} - MISC"
        
        messy_rows.append(messy_row)
        
    # Reconstruct DataFrame with the drifted schemas (resulting in lots of NaNs)
    return pd.DataFrame(messy_rows)

# Generate and save the data
print("Generating clean base data...")
clean_df = generate_base_data(10000)

print("Injecting schema drift and missing values...")
messy_df = inject_messiness(clean_df)

# Save to CSV
messy_df.to_csv('raw_claims_sprawl.csv', index=False)
print(f"Generated {len(messy_df)} records across {len(messy_df.columns)} drifting columns.")
print("Preview of the chaos:")
print(messy_df.sample(5))
