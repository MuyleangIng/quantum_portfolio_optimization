import random
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Database configuration
POSTGRES_PORT = 5432
POSTGRES_USER = "your_username"
POSTGRES_PASSWORD = "your_password"
POSTGRES_DB = "quantum_portfolio"
POSTGRES_SERVER = "localhost"

SQLALCHEMY_DATABASE_URI = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_SERVER}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Set up SQLAlchemy engine and session
engine = create_engine(SQLALCHEMY_DATABASE_URI)
Session = sessionmaker(bind=engine)
session = Session()

# Define base model
Base = declarative_base()

# Define stock_data table structure
class StockData(Base):
    __tablename__ = 'stock_data'
    
    id = Column(Integer, primary_key=True)
    date = Column(Date)
    stock = Column(String)
    price = Column(Float)
    volume = Column(Integer)

# Create table if it doesn't exist
Base.metadata.create_all(engine)

# Stock data generation parameters
stocks = ['NVDA', 'MSFT', 'GOOGL', 'AMZN', 'AAPL', 'TSLA']
start_date = datetime(2020, 1, 1)

# Generate and insert 1000 rows of stock data
for i in range(1000):
    date = start_date + timedelta(days=random.randint(0, 1000))
    stock = random.choice(stocks)
    price = round(random.uniform(300, 800), 2)
    volume = random.randint(100000, 1000000)
    
    # Create new stock data entry
    stock_entry = StockData(date=date, stock=stock, price=price, volume=volume)
    
    # Add to session
    session.add(stock_entry)

# Commit the transaction to insert all data
session.commit()

print("Inserted 1000 rows of stock data successfully.")
