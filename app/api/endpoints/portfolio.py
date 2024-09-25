from fastapi import APIRouter, Depends, UploadFile, File
from sqlalchemy.orm import Session
from app.db.session import get_db
import pandas as pd
from io import StringIO

router = APIRouter()

@router.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')))
    df.to_sql('stock_data', db.bind, if_exists='append', index=False)
    return {"message": f"Uploaded {len(df)} rows of data"}

@router.get("/export-csv")
def export_csv(db: Session = Depends(get_db)):
    query = "SELECT * FROM stock_data LIMIT 1000"  # Adjust as needed
    df = pd.read_sql(query, db.bind)
    csv_content = df.to_csv(index=False)
    return {"csv_content": csv_content}