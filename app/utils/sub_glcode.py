from sqlalchemy.ext.asyncio import AsyncSession
from app.models.sub_glcodes import SubGLCodes
from fastapi import UploadFile
from app.routes.csv_routes import excel_to_records, read_excel_file


REQUIRED_HEADERS = ["SubGLCode", "SubHeadID", "SubHeadName", "Head", "HeadID"]

async def create_sub_glcode(db: AsyncSession, data: UploadFile):
    """Create a new Sub GL Code entry in the database from uploaded Excel file"""
    if not data.filename.endswith((".xlsx", ".xls")):
        raise ValueError("Invalid file format. Please upload an Excel file.")
    
    df = await read_excel_file(data)
    records = excel_to_records(df, REQUIRED_HEADERS)

    if not records:
        raise ValueError("No records found in the uploaded file.")

    # Create ORM objects
    objs = [SubGLCodes(**rec) for rec in records]

    # Bulk insert
    db.add_all(objs)
    await db.commit()

    # Refresh only if you need objects back
    for obj in objs:
        await db.refresh(obj)

    return objs
