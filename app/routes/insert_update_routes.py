import logging
from sqlalchemy import select
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field, validator
from pathlib import Path

from app.utils.database import get_db
from app.models import CSVHeaders

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/records", tags=["Records Management"])
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))


class RecordRequest(BaseModel):
    """Schema for insert/update record request"""

    sub_gl_code: str = Field(
        ..., min_length=1, max_length=50, description="Sub GL Code"
    )
    sub_head: str = Field(
        ..., min_length=1, max_length=255, description="Sub Head description"
    )
    region: str = Field(..., min_length=1, max_length=100, description="Region")
    branch_code: str = Field(
        ..., min_length=1, max_length=50, description="Branch Code"
    )
    branch_name: str = Field(
        ..., min_length=1, max_length=255, description="Branch Name"
    )
    record_id: Optional[int] = Field(None, description="Record ID for updates")

    @validator("sub_gl_code", "branch_code")
    def validate_alphanumeric(cls, v):
        import re

        if not re.match(r"^[A-Za-z0-9\-_]+$", v):
            raise ValueError(
                "Must contain only alphanumeric characters, hyphens, and underscores"
            )
        return v

    class Config:
        schema_extra = {
            "example": {
                "sub_gl_code": "GL001",
                "sub_head": "Office Supplies",
                "region": "North",
                "branch_code": "BR001",
                "branch_name": "Main Branch",
                "record_id": None,
            }
        }


class RecordResponse(BaseModel):
    """Schema for record response"""

    success: bool
    message: str
    record_id: Optional[int] = None


@router.get("/insert-update", response_class=HTMLResponse)
async def insert_update_page(request: Request):
    """Render the insert/update records page"""
    return templates.TemplateResponse(
        "insert_update.html", {"request": request, "title": "Insert/Update Records"}
    )


@router.get("/api/check-record")
async def check_existing_record(
    sub_gl_code: str, branch_code: str, db: AsyncSession = Depends(get_db)
):
    """Check if a record exists based on sub_gl_code and branch_code"""
    try:
        if not sub_gl_code or not branch_code:
            raise HTTPException(
                status_code=400, detail="Both sub_gl_code and branch_code are required"
            )

        result = await db.execute(
            select(CSVHeaders).filter(
                CSVHeaders.sub_gl_code == sub_gl_code.strip(),
                CSVHeaders.branch_code == branch_code.strip(),
            )
        )
        record = result.scalar_one_or_none()

        if record:
            return {
                "exists": True,
                "record": {
                    "id": record.id,
                    "sub_gl_code": record.sub_gl_code,
                    "sub_head": record.sub_head,
                    "region": record.region,
                    "branch_code": record.branch_code,
                    "branch_name": record.branch_name,
                },
            }
        else:
            return {"exists": False}

    except Exception as e:
        logger.error(f"Error checking existing record: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/insert-update-record", response_model=RecordResponse)
async def insert_update_record(request: RecordRequest, db: AsyncSession = Depends(get_db)):
    """Insert a new record or update an existing one"""
    try:
        data = request.dict()

        # Clean data
        for key in ["sub_gl_code", "sub_head", "region", "branch_code", "branch_name"]:
            if key in data and data[key]:
                data[key] = data[key].strip()

        record_id = data.pop("record_id", None)

        if record_id:
            # Update existing record
            result = await db.execute(
                select(CSVHeaders).filter(CSVHeaders.id == record_id)
            )
            record = result.scalar_one_or_none()

            if not record:
                raise HTTPException(status_code=404, detail="Record not found")

            # Update fields
            record.sub_gl_code = data["sub_gl_code"]
            record.sub_head = data["sub_head"]
            record.region = data["region"]
            record.branch_code = data["branch_code"]
            record.branch_name = data["branch_name"]

            db.commit()

            return RecordResponse(
                success=True, message="Record updated successfully", record_id=record.id
            )
        else:
            # Insert new record
            # Check if record already exists
            result = await db.execute(
                select(CSVHeaders).filter(
                    CSVHeaders.sub_gl_code == data["sub_gl_code"],
                    CSVHeaders.branch_code == data["branch_code"],
                )
            )
            existing_record = result.scalar_one_or_none()

            if existing_record:
                raise HTTPException(
                    status_code=409,
                    detail="A record with this Sub GL Code and Branch Code already exists",
                )

            new_record = CSVHeaders(**data)
            db.add(new_record)
            db.commit()
            db.refresh(new_record)

            return RecordResponse(
                success=True,
                message="Record inserted successfully",
                record_id=new_record.id,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing record: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/records/{record_id}")
async def get_record(record_id: int, db: AsyncSession = Depends(get_db)):
    """Get a single record by ID"""
    try:
        result = await db.execute(select(CSVHeaders).filter(CSVHeaders.id == record_id))
        record = result.scalar_one_or_none()

        if not record:
            raise HTTPException(status_code=404, detail="Record not found")

        return {
            "success": True,
            "record": {
                "id": record.id,
                "sub_gl_code": record.sub_gl_code,
                "sub_head": record.sub_head,
                "region": record.region,
                "branch_code": record.branch_code,
                "branch_name": record.branch_name,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving record: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/records/{record_id}", response_model=RecordResponse)
async def delete_record(record_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a record by ID"""
    try:
        record = await db.execute(select(CSVHeaders).filter(CSVHeaders.id == record_id))
        record = record.scalar_one_or_none()

        if not record:
            raise HTTPException(status_code=404, detail="Record not found")

        db.delete(record)
        db.commit()

        return RecordResponse(success=True, message="Record deleted successfully")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting record: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
