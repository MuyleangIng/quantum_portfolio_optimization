import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from app.core.config import settings

# Create a database engine
engine = create_engine(settings.SQLALCHEMY_DATABASE_URI)

# Generate sample data
dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
stocks = ['AAPL', 'AMZN', 'FB', 'GOOGL', 'JNJ', 'JPM', 'MSFT', 'NVDA', 'TSLA', 'V']

data = []
for date in dates:
    for stock in stocks:
        price = np.random.uniform(100, 1000)  # Random price between 100 and 1000
        volume = np.random.randint(1000, 1000000)  # Random volume between 1000 and 1000000
        data.append({'date': date, 'stock': stock, 'price': price, 'volume': volume})

df = pd.DataFrame(data)

# Save to database
df.to_sql('stock_data', engine, if_exists='replace', index=False)

print(f"Generated {len(df)} rows of sample data.")