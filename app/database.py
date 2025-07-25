import pandas as pd

def load_products_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")