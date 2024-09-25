from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from app.db.session import get_db
from sqlalchemy import text
from qiskit_finance.exceptions import QiskitFinanceError
from qiskit_optimization.algorithms import MinimumEigenOptimizer

import pandas as pd
from io import StringIO
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit.primitives import Sampler
import numpy as np
import logging
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from app.db.session import get_db
router = APIRouter()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# @router.post("/upload-csv")
# async def upload_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
#     contents = await file.read()
#     df = pd.read_csv(StringIO(contents.decode('utf-8')))
    
#     try:
#         # Ensure the DataFrame has the expected columns
#         required_columns = ['date', 'stock', 'price', 'volume']
#         if not all(col in df.columns for col in required_columns):
#             raise ValueError(f"CSV must contain columns: {', '.join(required_columns)}")
        
#         # Convert date to datetime
#         df['date'] = pd.to_datetime(df['date'])
        
#         # Save to database
#         df.to_sql('stock_data', db.bind, if_exists='append', index=False)
        
#         return {"message": f"Uploaded {len(df)} rows of data"}
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))
@router.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        logger.info(f"Starting CSV upload process for file: {file.filename}")
        
        contents = await file.read()
        logger.info(f"File contents read. Size: {len(contents)} bytes")
        
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        logger.info(f"CSV parsed into DataFrame. Shape: {df.shape}")
        
        # Ensure the DataFrame has the expected columns
        required_columns = ['date', 'stock', 'price', 'volume']
        if not all(col in df.columns for col in required_columns):
            missing_columns = set(required_columns) - set(df.columns)
            logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(f"CSV must contain columns: {', '.join(required_columns)}")
        
        logger.info("All required columns present in CSV")
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        logger.info("Date column converted to datetime")
        
        # Log a sample of the data
        logger.info(f"Sample data:\n{df.head().to_string()}")
        
        # Save to database
        logger.info("Attempting to save data to database")
        df.to_sql('stock_data', db.bind, if_exists='append', index=False)
        logger.info(f"Successfully saved {len(df)} rows to database")
        
        # Verify data was saved
        verify_query = text("SELECT COUNT(*) FROM stock_data")
        result = db.execute(verify_query)
        count = result.scalar()
        logger.info(f"Total rows in stock_data table after upload: {count}")
        
        return {"message": f"Uploaded {len(df)} rows of data. Total rows in database: {count}"}
    
    except pd.errors.EmptyDataError:
        logger.error("The CSV file is empty")
        raise HTTPException(status_code=400, detail="The CSV file is empty")
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error parsing CSV: {str(e)}")
    except ValueError as e:
        logger.error(str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during CSV upload: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@router.get("/check-data")
def check_data(db: Session = Depends(get_db)):
    try:
        query = text("SELECT COUNT(*) FROM stock_data")
        result = db.execute(query)
        count = result.scalar()
        return {"message": f"Total rows in stock_data table: {count}"}
    except Exception as e:
        logger.error(f"Error checking data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error checking data: {str(e)}")

@router.post("/optimize-portfolio")
def optimize_portfolio(db: Session = Depends(get_db)):
    try:
        # Fetch data from the database
        query = "SELECT * FROM stock_data ORDER BY date DESC LIMIT 1000"
        df = pd.read_sql(query, db.bind)
        
        print("Initial data shape:", df.shape)
        print("Initial data columns:", df.columns)
        print("Initial data sample:")
        print(df.head())
        
        # Ensure we have data
        if df.empty:
            raise HTTPException(status_code=400, detail="No data available for optimization")
        
        # Convert date to datetime if it's not already
        df['date'] = pd.to_datetime(df['date'])
        
        # Group by date and stock, taking the last price of the day if there are duplicates
        df = df.groupby(['date', 'stock'])['price'].last().reset_index()
        
        # Pivot the dataframe to have dates as index and stocks as columns
        df_pivot = df.pivot(index='date', columns='stock', values='price')
        
        print("Pivoted data shape:", df_pivot.shape)
        print("Pivoted data sample:")
        print(df_pivot.head())
        
        # Calculate returns
        returns = df_pivot.pct_change().dropna()
        
        print("Returns data shape:", returns.shape)
        print("Returns data sample:")
        print(returns.head())
        
        # Calculate expected returns (mean of returns for each stock)
        expected_returns = returns.mean()
        
        # Calculate covariance matrix
        cov_matrix = returns.cov()
        
        print("Expected returns:")
        print(expected_returns)
        print("\nCovariance matrix sample:")
        print(cov_matrix.iloc[:5, :5])
        
        # Ensure expected_returns and cov_matrix have the same order of stocks
        common_stocks = sorted(set(expected_returns.index) & set(cov_matrix.index))
        expected_returns = expected_returns[common_stocks]
        cov_matrix = cov_matrix.loc[common_stocks, common_stocks]
        
        # Convert to numpy arrays
        expected_returns_array = expected_returns.values
        cov_matrix_array = cov_matrix.values
        
        print("Expected returns shape:", expected_returns_array.shape)
        print("Covariance matrix shape:", cov_matrix_array.shape)
        
        # Set up the portfolio optimization problem
        num_assets = len(common_stocks)
        q = 0.5  # Risk factor
        budget = num_assets // 2  # Invest in half of the assets
        
        portfolio = PortfolioOptimization(
            expected_returns=expected_returns_array, 
            covariances=cov_matrix_array, 
            risk_factor=q, 
            budget=budget
        )
        qp = portfolio.to_quadratic_program()
        
        # Set up QAOA
        optimizer = COBYLA()
        sampler = Sampler()
        qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=3)
        
        # Use MinimumEigenOptimizer to run QAOA
        min_eigen_optimizer = MinimumEigenOptimizer(qaoa)
        result = min_eigen_optimizer.solve(qp)
        
        # Process results
        x = portfolio.interpret(result.x)
        selected_assets = [stock for stock, weight in zip(common_stocks, x) if weight > 0.01]
        weights = [weight for weight in x if weight > 0.01]
        
        return {
            "selected_assets": selected_assets,
            "weights": weights,
            "objective_value": result.fval
        }
    
    except QiskitFinanceError as e:
        print(f"QiskitFinanceError: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in portfolio optimization: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# Add this at the end of your file to print version information
import qiskit
import qiskit_finance

print("Qiskit version:", qiskit.__version__)
print("Qiskit Finance version:", qiskit_finance.__version__)
# @router.post("/optimize-portfolio")
# def optimize_portfolio(db: Session = Depends(get_db)):
#     # Fetch data from the database
#     query = "SELECT * FROM stock_data ORDER BY date DESC LIMIT 1000"
#     df = pd.read_sql(query, db.bind)
    
#     # Print initial data info
#     print("Initial data shape:", df.shape)
#     print("Initial data columns:", df.columns)
#     print("Initial data sample:")
#     print(df.head())
    
#     # Ensure we have data
#     if df.empty:
#         raise HTTPException(status_code=400, detail="No data available for optimization")
    
#     # Convert date to datetime if it's not already
#     df['date'] = pd.to_datetime(df['date'])
    
#     # Check for any invalid price values
#     if df['price'].isnull().any() or (df['price'] <= 0).any():
#         print("Invalid price values found:")
#         print(df[df['price'].isnull() | (df['price'] <= 0)])
#         raise HTTPException(status_code=400, detail="Invalid price values found in the data")
    
#     # Group by date and stock, taking the last price of the day if there are duplicates
#     df = df.groupby(['date', 'stock'])['price'].last().reset_index()
    
#     # Pivot the dataframe to have dates as index and stocks as columns
#     df_pivot = df.pivot(index='date', columns='stock', values='price')
    
#     # Sort by date to ensure chronological order
#     df_pivot = df_pivot.sort_index()
    
#     # Print pivoted data info
#     print("Pivoted data shape:", df_pivot.shape)
#     print("Pivoted data sample:")
#     print(df_pivot.head())
    
#     # Calculate returns
#     returns = df_pivot.pct_change().dropna()
    
#     # Print returns info
#     print("Returns data shape:", returns.shape)
#     print("Returns data sample:")
#     print(returns.head())
    
#     # Calculate expected returns (mean of returns for each stock)
#     expected_returns = returns.mean()
    
#     # Calculate covariance matrix
#     cov_matrix = returns.cov()
    
#     # Print expected returns and covariance matrix info
#     print("Expected returns:")
#     print(expected_returns)
#     print("\nCovariance matrix sample:")
#     print(cov_matrix.iloc[:5, :5])  # Print just a 5x5 sample
    
#     # Check for NaN values
#     if expected_returns.isnull().any() or cov_matrix.isnull().any().any():
#         print("NaN values found in expected returns or covariance matrix")
#         raise HTTPException(status_code=400, detail="NaN values found in calculated returns or covariances")
    
#     # Ensure expected_returns and cov_matrix have the same stocks
#     common_stocks = list(set(expected_returns.index) & set(cov_matrix.index))
#     expected_returns = expected_returns[common_stocks]
#     cov_matrix = cov_matrix.loc[common_stocks, common_stocks]
    
#     # Print final data for optimization
#     print(f"Number of assets: {len(common_stocks)}")
#     print("Final expected returns:")
#     print(expected_returns)
#     print("\nFinal covariance matrix sample:")
#     print(cov_matrix.iloc[:5, :5])  # Print just a 5x5 sample
    
#     # Set up the portfolio optimization problem
#     num_assets = len(common_stocks)
#     q = 0.5  # Risk factor
#     budget = num_assets // 2  # Invest in half of the assets
    
#     portfolio = PortfolioOptimization(expected_returns=expected_returns, covariances=cov_matrix, risk_factor=q, budget=budget)
#     qp = portfolio.to_quadratic_program()
    
#     # Set up QAOA
#     optimizer = COBYLA()
#     sampler = Sampler()
#     qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=3)
    
#     # Run QAOA
#     result = qaoa.optimize(qp)
    
#     # Process results
#     x = portfolio.interpret(result.x)
#     selected_assets = [stock for stock, weight in zip(common_stocks, x) if weight > 0.01]
#     weights = [weight for weight in x if weight > 0.01]
    
#     return {
#         "selected_assets": selected_assets,
#         "weights": weights,
#         "objective_value": result.fval
#     }
@router.get("/export-csv-tolocal")
def export_csv(db: Session = Depends(get_db)):
    query = "SELECT * FROM stock_data LIMIT 1000"
    df = pd.read_sql(query, db.bind)
    
    # Specify the local file path where you want to save the CSV
    file_path = "/Users/sen/Developer/quantum/quantum_portfolio_optimization/data/test.csv"  # Update this path
    
    # Save the DataFrame as a CSV file on the server (localhost)
    df.to_csv(file_path, index=False)
    
    return {"message": f"CSV file saved at {file_path}"}
@router.get("/test-db")
def test_db(db: Session = Depends(get_db)):
    try:
        # Use text() to wrap the SQL string
        result = db.execute(text("SELECT 1"))
        result.scalar()  # Fetch the result to ensure the query was executed
        return {"message": "Database connection successful"}
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")
