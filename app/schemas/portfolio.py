from pydantic import BaseModel
from typing import List, Dict

class PortfolioOptimizationRequest(BaseModel):
    tickers: List[str]
    start_date: str
    end_date: str
    risk_factor: float
    budget: float

class PortfolioOptimizationResponse(BaseModel):
    allocations: Dict[str, float]
