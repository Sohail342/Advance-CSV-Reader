from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from app.utils.database import get_db

from app.utils.sub_glcode import create_sub_glcode

router = APIRouter(prefix="/api", tags=["Sub GL Codes"])

@router.post("/sub_glcodes/", status_code=201)
async def add_sub_glcode(sub_glcode_data: UploadFile, db: AsyncSession = Depends(get_db)):
    try:
        new_sub_glcode = await create_sub_glcode(db, sub_glcode_data)
        return {"message": "Sub GL Code created successfully", "data": new_sub_glcode}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))