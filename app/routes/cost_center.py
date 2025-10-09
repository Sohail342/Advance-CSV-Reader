from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from app.utils.database import get_db

from app.utils.cost_center import create_cost_center

router = APIRouter(prefix="/api", tags=["Cost Centers"])

@router.post("/create_cost/", status_code=201)
async def add_cost_center(cost_center_data: UploadFile, db: AsyncSession = Depends(get_db)):
    try:
        new_cost_center = await create_cost_center(db, cost_center_data)
        return {"message": "CostCenter created successfully", "data": new_cost_center}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))