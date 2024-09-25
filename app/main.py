from fastapi import FastAPI
from app.api.endpoints import data  # This imports your router

app = FastAPI(
    title="Quantum Portfolio Optimization API",
    description="API for quantum portfolio optimization with CSV upload and export capabilities",
    version="1.0.0",
)

app.include_router(data.router, prefix="/api/v1", tags=["data"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)