from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models import CSVHeaders
from fastapi import HTTPException


async def get_new_records(
    columns_missed: bool, SubGLCode: str, CostCenterID: str, db: AsyncSession
) -> list:
    """Get new columns from database to add in csv if columns are missed."""
    if not columns_missed:
        return []

    try:
        result = await db.execute(
            select(
                CSVHeaders.SubGLCode,
                CSVHeaders.CostCenterID,
                CSVHeaders.SubHeadID,
                CSVHeaders.SubHead,
                CSVHeaders.Region,
                CSVHeaders.BCode,
                CSVHeaders.BName,
                CSVHeaders.BudgetID,
                CSVHeaders.BudgetAmount,
                CSVHeaders.Head,
                CSVHeaders.HeadID,
                CSVHeaders.CostCenter,
                CSVHeaders.ValidityDate,
                CSVHeaders.Description,
            ).where(
                CSVHeaders.SubGLCode == SubGLCode,
                CSVHeaders.CostCenterID == CostCenterID,
            )
        )

        rows = result.all()
        if not rows:
            raise HTTPException(status_code=404, detail="No new records found.")

        new_records_list = [dict(row._mapping) for row in rows]

        return new_records_list
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching new records: {str(e)}"
        )