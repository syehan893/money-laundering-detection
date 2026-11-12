import pandas as pd

# Load the dataset from Hugging Face
file_path = "hf://datasets/Pcitycrypto/xauusd/XAUUSD_1m.csv"
df = pd.read_csv(file_path, parse_dates=['datetime'], index_col='datetime')

# Display the first few rows
print(df.head())
