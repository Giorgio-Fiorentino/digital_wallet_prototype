import sys
import os
import pandas as pd

print("PYTHON:", sys.executable)
csv_path = "data/transactions.csv"
if not os.path.exists(csv_path):
	print("File not found:", csv_path)
	sys.exit(1)

dataset = pd.read_csv(csv_path)
# In a script you must print the result explicitly; calling dataset.head()
# alone only returns a DataFrame in interactive shells and produces no output.
print(dataset.head().to_string(index=False))
print(f"\nLoaded {len(dataset)} rows and {len(dataset.columns)} columns from {csv_path}")