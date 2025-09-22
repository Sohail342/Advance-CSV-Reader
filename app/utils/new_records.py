from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models import CSVHeaders
from fastapi import HTTPException


async def get_new_records(columns_missed: bool, SubGLCode: str = None, BCode: str = None, db: AsyncSession = None):
    out_of_scop = []
    if not columns_missed:
        return [], []

    try:
        q = select(
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
        )
        if SubGLCode and BCode:
            q = q.where(
                CSVHeaders.SubGLCode == SubGLCode,
                CSVHeaders.BCode == BCode,
            )
        elif SubGLCode:
            q = q.where(CSVHeaders.SubGLCode == SubGLCode)

        result = await db.execute(q)
        rows = result.all()

        if not rows:
            # out of scope → return SubGLCode for reporting
            print(SubGLCode)
            out_of_scop.append(SubGLCode)

        new_records_list = [dict(row._mapping) for row in rows]
        return new_records_list, out_of_scop
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching new records: {str(e)}"
        )
