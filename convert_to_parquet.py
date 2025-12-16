

import pandas as pd

# 🚨 CHANGE THIS TO MATCH YOUR ACTUAL CSV FILE NAME 🚨
CSV_FILENAME = 'spotify_millsongdata.csv' 
PARQUET_FILENAME = 'spotify_millsongdata.parquet'

try:
    df = pd.read_csv(CSV_FILENAME)
    df.to_parquet(PARQUET_FILENAME, index=False)
    print(f"SUCCESS: Data converted and saved to {PARQUET_FILENAME}")
except FileNotFoundError:
    print(f"FATAL ERROR: CSV file '{CSV_FILENAME}' not found. Please check the name.")
except Exception as e:
    print(f"An error occurred during conversion. Did you install PyArrow? Error: {e}")