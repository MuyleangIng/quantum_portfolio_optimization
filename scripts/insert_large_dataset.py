import pandas as pd
from sqlalchemy import create_engine, text
from app.core.config import settings

def export_data_to_csv(query, csv_path):
    engine = create_engine(settings.SQLALCHEMY_DATABASE_URI)
    
    with engine.connect() as connection:
        df = pd.read_sql(query, connection)
        df.to_csv(csv_path, index=False)
        print(f"Data exported to {csv_path}")

if __name__ == "__main__":
    query = "SELECT * FROM stock_data LIMIT 1000000"  # Adjust this query as needed
    csv_path = 'data/processed/exported_stock_data.csv'  # Update this path as needed
    export_data_to_csv(query, csv_path)