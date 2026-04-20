
import kagglehub
import pandas as pd
import os

# Download dataset
path = kagglehub.dataset_download("dhairyajeetsingh/ecommerce-customer-behavior-dataset")

print(f"Dataset downloaded to: {path}")

# List files in the path
files = os.listdir(path)
print(f"Files in dataset: {files}")

# Load the main CSV file (assuming there's one)
for file in files:
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(path, file))
        print(f"\n--- {file} ---")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Head:\n{df.head()}")
        print(f"Info:\n")
        df.info()
