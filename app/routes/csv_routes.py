import os
import csv
import io
import tempfile
import logging
from uuid import uuid4
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from sqlalchemy import tuple_, select

import pandas as pd
from fastapi import (
    APIRouter,
    Depends,
    File,
    UploadFile,
    HTTPException,
    Request,
    BackgroundTasks,
    Query as Q,
)
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
from app.schemas import CSVRecordOut

from app.utils.new_records import get_new_records
from app.models import CSVHeaders
from app.utils.database import get_db
from app.utils.excel_processor import (
    read_excel_file,
    validate_excel_headers,
    excel_to_records,
    EXCEL_EXTENSIONS,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/csv", tags=["CSV Processing"])
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))

columns_for_matched_records = [
    "SubGLCode", "CostCenterID", "SubHeadID", "SubHead", "Region", "BCode", "BName",
    "Head", "HeadID", "CostCenter", 
    "ValidityDate", "Description", "BudgetID", "BudgetAmount", "NewBudget", "NewOldAmountComparison"
]

columns_for_unmatched_records = [
    "SubGLCode", "CostCenterID", "SubHeadID", "SubHead", "Region", "BCode", "BName",
    "Head", "HeadID", "CostCenter", 
    "ValidityDate", "Description", "BudgetID", "BudgetAmount",
]

columns_for_matched_records_ready_to_upload = [
    "BudgetID", "HeadID", "Head", "SubHeadID", "SubHead", "CostCenterID", "CostCenter",
    "BudgetAmount", "ValidityDate", "Description", "NewBudget", "NewOldAmountComparison"
]

REQUIRED_HEADERS = [
    "SubGLCode",
    "CostCenterID",
    "SubHeadID",
    "SubHead",
    "Region",
    "BCode",
    "BName",
    "BudgetID",
    "Head",
    "HeadID",
    "CostCenter",
    "BudgetAmount",
    "ValidityDate",
    "Description",
    "RemainingBudget",
]

csv_store: Dict[str, Dict[str, Optional[str]]] = {}


@router.get("/check-duplicate", response_class=HTMLResponse)
async def check_duplicate_page(request: Request):
    return templates.TemplateResponse(
        "check_duplicate.html", {"request": request, "title": "Check Duplicates"}
    )


@router.post("/check_duplicate", response_class=HTMLResponse)
async def check_duplicate_file(
    request: Request,
    file: UploadFile = File(...),
    remove: bool = Q(False, description="Set to true to remove duplicates"),
    db: AsyncSession = Depends(get_db),
):
    fname = file.filename.lower()
    ext = os.path.splitext(fname)[1]

    if ext in EXCEL_EXTENSIONS:
        df = await read_excel_file(file)
        records = excel_to_records(df, REQUIRED_HEADERS)
    else:
        content = await file.read()
        ts = io.StringIO(content.decode("utf-8", errors="replace"))
        reader = csv.DictReader(ts)
        records = [
            {h: row.get(h, "").strip() for h in REQUIRED_HEADERS} for row in reader
        ]

    if not records:
        return templates.TemplateResponse(
            "check_duplicate.html",
            {
                "request": request,
                "title": "Check Duplicates",
                "error": "No valid records found in the file",
            },
        )
    

    duplicates = []
    unique_set = set()
    unique_records = []
    for i, rec in enumerate(records):
        key = (rec["SubGLCode"], rec["CostCenterID"])
        if key not in unique_set:
            unique_set.add(key)
            unique_records.append(rec)
        else:
            duplicates.append(
                {
                    "row": i,
                    "SubGLCode": rec["SubGLCode"],
                    "CostCenterID": rec["CostCenterID"],
                    "SubHeadID": rec["SubHeadID"],
                    "SubHead": rec["SubHead"],
                    "Region": rec["Region"],
                    "BCode": rec["BCode"],
                    "BName": rec["BName"],
                    "BudgetID": rec["BudgetID"],
                    "Head": rec["Head"],
                    "HeadID": rec["HeadID"],
                    "CostCenter": rec["CostCenter"],
                    "BudgetAmount": rec["BudgetAmount"],
                    "ValidityDate": rec["ValidityDate"],
                    "Description": rec["Description"],
                }
            )
    if remove:
        # Generate cleaned file for download
        cleaned_df = pd.DataFrame(unique_records)

        # Create temporary file for download
        temp_file = tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False)
        cleaned_df.to_csv(temp_file.name, index=False)
        temp_file.close()

        return templates.TemplateResponse(
            "check_duplicate.html",
            {
                "request": request,
                "title": "Duplicates Removed",
                "message": f"Successfully removed {len(duplicates)} duplicate records.",
                "cleaned_records": unique_records,
                "cleaned_file_path": temp_file.name,
                "original_filename": file.filename,
                "duplicates_removed": len(duplicates),
                "total_original_records": len(records),
                "cleaned_records_count": len(unique_records),
            },
        )
    else:
        # Convert duplicates to dataframe
        duplicates_csv = pd.DataFrame(duplicates)

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False)
        duplicates_csv.to_csv(temp_file.name, index=False)
        temp_file.close()

        return templates.TemplateResponse(
            "check_duplicate.html",
            {
                "request": request,
                "title": "Check Duplicates",
                "duplicates": duplicates,
                "message": f"Found {len(duplicates)} duplicate records"
                if duplicates
                else "No duplicate records found",
                "total_records": len(records),
                "download_link": f"/csv/duplicate/download/{os.path.basename(temp_file.name)}"
                if duplicates
                else None,
            },
        )


@router.get("/duplicate/download/{filename}", response_class=FileResponse)
async def download_file(filename: str):
    try:
        file_path = os.path.join(tempfile.gettempdir(), filename)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": f"An error occurred while processing your file: {str(e)}"
            },
        )
    return FileResponse(file_path, filename=filename, media_type="text/csv")


@router.post("/remove_duplicates")
async def remove_duplicate_records_file(
    file: UploadFile = File(...),
):
    """
    API-style handler that removes duplicate records and returns the cleaned file directly.
    """
    try:
        fname = file.filename.lower()
        ext = os.path.splitext(fname)[1]

        if ext not in EXCEL_EXTENSIONS and ext != ".csv":
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Invalid file format. Please upload CSV or Excel files only."
                },
            )

        # Process file
        if ext in EXCEL_EXTENSIONS:
            df = await read_excel_file(file)
            records = excel_to_records(df, REQUIRED_HEADERS)
        else:
            content = await file.read()
            ts = io.StringIO(content.decode("utf-8", errors="replace"))
            reader = csv.DictReader(ts)
            records = [
                {h: row.get(h, "").strip() for h in REQUIRED_HEADERS} for row in reader
            ]

        if not records:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "No valid records found in the file. Please check your file format and required headers."
                },
            )

        # Deduplicate
        seen = set()
        unique_records = []
        for record in records:
            key = (record["SubGLCode"], record["CostCenterID"])
            if key not in seen:
                seen.add(key)
                unique_records.append(record)

        cleaned_df = pd.DataFrame(unique_records)

        # Create secure temp file
        temp_dir = tempfile.mkdtemp()
        cleaned_filename = f"cleaned_{file.filename}"
        cleaned_path = os.path.join(temp_dir, cleaned_filename)

       
        cleaned_df.to_csv(cleaned_path, index=False)
       

        # Return cleaned file directly
        return FileResponse(
            path=cleaned_path,
            filename=cleaned_filename,
            media_type="text/csv"
        )

    except Exception as e:
        logger.error(f"Error removing duplicates: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": f"An error occurred while processing your file: {str(e)}"
            },
        )


@router.get("/download_cleaned")
async def download_cleaned_file(
    file: str, filename: str, background_tasks: BackgroundTasks
):
    """
    Secure endpoint for downloading cleaned files.
    Automatically cleans up temporary files after download.
    """
    try:
        # Security check: ensure file exists and is in temp directory
        file_path = Path(file)
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        # Ensure file is in a temporary directory
        if "temp" not in str(file_path.parent).lower():
            raise HTTPException(status_code=403, detail="Invalid file location")

        # Schedule file cleanup after download
        background_tasks.add_task(cleanup_file, file_path)
        background_tasks.add_task(cleanup_directory, file_path.parent)

        return FileResponse(
            path=file_path, filename=filename, media_type="application/octet-stream"
        )

    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise HTTPException(status_code=500, detail="Error downloading file")


def cleanup_file(file_path: Path):
    """Clean up temporary file."""
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        logger.error(f"Error cleaning up file {file_path}: {str(e)}")


def cleanup_directory(dir_path: Path):
    """Clean up temporary directory if empty."""
    try:
        if dir_path.exists() and dir_path.is_dir():
            if not any(dir_path.iterdir()):
                dir_path.rmdir()
    except Exception as e:
        logger.error(f"Error cleaning up directory {dir_path}: {str(e)}")


@router.get("/search_existing", response_model=List[CSVRecordOut])
async def search_existing_records(
    sub_gl_codes: List[str] = Q(...),
    branch_codes: List[str] = Q(...),
    db: AsyncSession = Depends(get_db),
):
    if not sub_gl_codes or not branch_codes:
        raise HTTPException(
            status_code=400, detail="sub_gl_codes and branch_codes are required"
        )

    try:
        stmt = select(CSVHeaders).where(
            CSVHeaders.SubGLCode.in_(sub_gl_codes),
            CSVHeaders.BCode.in_(branch_codes),
        )
        result = await db.execute(stmt)
        records = result.scalars().all()
        if not records:
            raise HTTPException(status_code=404, detail="No matching records found")
        return records
    except SQLAlchemyError as e:
        logger.error(f"Database error during search: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


def save_df_to_temp_file(df: pd.DataFrame, suffix: str = ".csv") -> str:
    temp = tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix, mode="w", newline="", encoding="utf-8"
    )
    df.to_csv(temp.name, index=False)
    temp.close()
    return temp.name


async def process_batch(
    batch: List[Dict[str, str]], db: AsyncSession, stats: Dict[str, Any]
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    # Extract lookup pairs
    lookup_pairs = [
        (rec["SubGLCode"], rec["BCode"])
        for rec in batch
        if rec.get("SubGLCode") and rec.get("BCode")
    ]

    stmt = select(CSVHeaders).where(
        tuple_(CSVHeaders.SubGLCode, CSVHeaders.BCode).in_(lookup_pairs)
    )
    result = await db.execute(stmt)
    existing_records = result.scalars().all()
    
    # Build a lookup map
    existing_map = {(str(e.SubGLCode), str(e.BCode)): e for e in existing_records}

    matched = []
    new_objs = []

    for rec in batch:
        key = (rec["SubGLCode"], rec["BCode"])
        if key in existing_map:
            stats["matched_records"] += 1
            matched.append(rec)
        else:
            stats["new_records"] += 1
            new_objs.append(rec)

    return (
        pd.DataFrame(matched) if matched else None,
        pd.DataFrame(new_objs) if new_objs else None,
    )

@router.get("/test/{code}")
async def text(code: str, db: AsyncSession = Depends(get_db)):
    records_from_db = await get_new_records(
                    columns_missed=True,
                    SubGLCode=code,
                    db=db,
                )
    return JSONResponse(records_from_db)

async def process_csv_file(
    file: UploadFile, db: AsyncSession
) -> Tuple[Dict[str, Any], Optional[str], Optional[str], Optional[str], Optional[str]]:
    stats = {
        "total_rows": 0,
        "matched_records": 0,
        "new_records": 0,
        "error_records": 0,
        "errors": [],
    }

    records = []
    fname = file.filename.lower()
    ext = os.path.splitext(fname)[1]

    # --- Read file into records ---
    if ext in EXCEL_EXTENSIONS:
        df = await read_excel_file(file)
        records = excel_to_records(df, REQUIRED_HEADERS)
    else:
        content = await file.read()
        ts = io.StringIO(content.decode("utf-8", errors="replace"))
        reader = csv.DictReader(ts)
        for i, row in enumerate(reader, start=2):
            stats["total_rows"] += 1
            if not row.get("SubGLCode"):
                stats["error_records"] += 1
                stats["errors"].append(f"Row {i}: Missing SubGLCode")
                continue
            records.append({h: row.get(h, "").strip() for h in REQUIRED_HEADERS})

    stats["total_rows"] = stats["total_rows"] or len(records)

    # --- Holders ---
    all_matched = []
    all_new = []
    all_matched_two = []
    all_out_of_scope = []

    df_matched = None
    df_new = None

    batch_size = 100
    for i in range(0, len(records), batch_size):
        batch = records[i: i + batch_size]
        df_matched, df_new = await process_batch(batch, db, stats)

    # --- Process Matched ---
    if df_matched is not None and not df_matched.empty:
        for _, row in df_matched.iterrows():
            records_from_db, out_of_scop = await get_new_records(
                columns_missed=True,
                SubGLCode=str(row["SubGLCode"]),
                BCode=str(row["BCode"]),
                db=db,
            )
            if records_from_db:
                for rec in records_from_db:
                    rec["NewBudget"] = row["RemainingBudget"]
                    rec["NewOldAmountComparison"] = (
                        "True" if str(row["RemainingBudget"]) == str(rec["BudgetAmount"]) else "False"
                    )

                df_db = pd.DataFrame(records_from_db)

                # Fill missing columns
                for col in columns_for_matched_records:
                    if col not in df_db:
                        df_db[col] = ""

                df_db = df_db[columns_for_matched_records]
                all_matched.append(df_db)

                # Format two
                if all(col in df_db.columns for col in columns_for_matched_records_ready_to_upload):
                    df_db_format_two = df_db[columns_for_matched_records_ready_to_upload]
                    all_matched_two.append(df_db_format_two)

    # --- Process New ---
    if df_new is not None and not df_new.empty:
        for _, row in df_new.iterrows():
            records_from_db, out_of_scope = await get_new_records(
                columns_missed=True,
                SubGLCode=str(row["SubGLCode"]),
                db=db,
            )

            if out_of_scope:
                all_out_of_scope.extend(out_of_scope)

            if records_from_db:

                df_db = pd.DataFrame(records_from_db)

                for col in columns_for_unmatched_records:
                    if col not in df_db:
                        df_db[col] = ""

                df_db = df_db[columns_for_unmatched_records]
                all_new.append(df_db)

    # --- Concatenate Helper ---
    def concat(dfs):
        return pd.concat(dfs, ignore_index=True) if dfs else None

    matched_df = concat(all_matched)
    matched_two_df = concat(all_matched_two)
    new_df = concat(all_new)
    out_of_scope_df = pd.DataFrame(all_out_of_scope, columns=["SubGLCode"])

    # --- Save to temp files ---
    matched_path = (
        save_df_to_temp_file(matched_df)
        if matched_df is not None and not matched_df.empty
        else None
    )
    matched_two_path = (
        save_df_to_temp_file(matched_two_df)
        if matched_two_df is not None and not matched_two_df.empty
        else None
    )
    new_path = (
        save_df_to_temp_file(new_df)
        if new_df is not None and not new_df.empty
        else None
    )
    out_of_scope_path = (
        save_df_to_temp_file(out_of_scope_df)
        if out_of_scope_df is not None and not out_of_scope_df.empty
        else None
    )

    return stats, matched_path, matched_two_path, new_path, out_of_scope_path



@router.get("/upload", response_class=HTMLResponse)
async def get_upload_form(request: Request):
    return templates.TemplateResponse(
        "upload.html", {"request": request, "title": "Upload CSV or Excel"}
    )


@router.post("/upload")
async def upload_data_file(
    request: Request,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    stats, matched_path, matched_two_path, new_path, out_of_scope_df = await process_csv_file(file, db)

    csv_id = str(uuid4())
    csv_store[csv_id] = {"matched": matched_path, "matched_ready_to_upload": matched_two_path, "out_of_scop_glcodes_path":out_of_scope_df, "new": new_path}

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "title": "CSV Results",
            "result": stats,
            "csv_id": csv_id,
            "operation_type": "original",
        },
    )


@router.post("/upload-comparison")
async def upload_or_insert_data(
    request: Request,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Handle comparison data upload that inserts new records and updates existing ones.
    Uses sub_gl_code and branch_code as unique identifiers for matching.
    """
    try:
        fname = file.filename.lower()
        ext = os.path.splitext(fname)[1]

        # Validate file type
        if ext not in EXCEL_EXTENSIONS and ext != ".csv":
            return templates.TemplateResponse(
                "result.html",
                {
                    "request": request,
                    "title": "Error",
                    "error": "Invalid file format. Please upload CSV or Excel files only.",
                    "result": {
                        "total_rows": 0,
                        "inserted_records": 0,
                        "updated_records": 0,
                        "error_records": 0,
                        "errors": ["Invalid file format"],
                    },
                    "csv_id": None,
                    "operation_type": "comparison",
                },
            )

        # Process file
        if ext in EXCEL_EXTENSIONS:
            df = await read_excel_file(file)
            # validate_excel_headers(df, REQUIRED_HEADERS)
            records = excel_to_records(df, REQUIRED_HEADERS)
        else:
            content = await file.read()
            ts = io.StringIO(content.decode("utf-8", errors="replace"))
            reader = csv.DictReader(ts)
            records = [
                {h: row.get(h, "").strip() for h in REQUIRED_HEADERS} for row in reader
            ]

        if not records:
            return templates.TemplateResponse(
                "result.html",
                {
                    "request": request,
                    "title": "Error",
                    "error": "No valid records found in the file. Please check your file format and required headers.",
                    "result": {
                        "total_rows": 0,
                        "inserted_records": 0,
                        "updated_records": 0,
                        "error_records": 0,
                        "errors": ["No valid records found"],
                    },
                    "csv_id": None,
                    "operation_type": "comparison",
                },
            )

        # Process comparison data - insert/update records
        stats = {
            "total_rows": len(records),
            "inserted_records": 0,
            "updated_records": 0,
            "error_records": 0,
            "errors": [],
        }

        inserted_records = []
        updated_records = []
        new_record = []

        # Process in batches for better performance
        batch_size = 100

        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]

            # Extract lookup pairs
            lookup_pairs = [
                (rec["SubGLCode"], rec["CostCenterID"])
                for rec in batch
                if rec.get("SubGLCode") and rec.get("CostCenterID")
            ]

            # Query existing records
            stmt = select(CSVHeaders).where(
                tuple_(CSVHeaders.SubGLCode, CSVHeaders.CostCenterID).in_(lookup_pairs)
            )
            result = await db.execute(stmt)
            existing_records = result.scalars().all()


            # Create lookup map
            existing_map = {(str(rec.SubGLCode), str(rec.CostCenterID)): rec for rec in existing_records}
            
            # Process each record in batch
            for record in batch:
                try:
                    key = (record["SubGLCode"], record["CostCenterID"])
                    if key in existing_map:
                        # Update existing record
                        existing_record = existing_map[key]
                        existing_record.SubGLCode = record["SubGLCode"]
                        existing_record.SubHeadID = record["SubHeadID"]
                        existing_record.SubHead = record["SubHead"]
                        existing_record.Region = record["Region"]
                        existing_record.BCode = record["BCode"]
                        existing_record.BName = record["BName"]
                        existing_record.BudgetID = record["BudgetID"]
                        existing_record.Head = record["Head"]
                        existing_record.HeadID = record["HeadID"]
                        existing_record.CostCenter = record["CostCenter"]
                        existing_record.BudgetAmount = record["BudgetAmount"]
                        existing_record.ValidityDate = record["ValidityDate"]
                        existing_record.Description = record["Description"]
                        updated_records.append(record)
                        stats["updated_records"] += 1
                    else:
                        # Insert new record
                        new_record.append(CSVHeaders(
                            CostCenterID=record["CostCenterID"],
                            SubHeadID=record["SubHeadID"],
                            SubGLCode=record["SubGLCode"],
                            SubHead=record["SubHead"],
                            Region=record["Region"],
                            BCode=record["BCode"],
                            BName=record["BName"],
                            BudgetID=record["BudgetID"],
                            Head=record["Head"],
                            HeadID=record["HeadID"],
                            CostCenter=record["CostCenter"],
                            BudgetAmount=record["BudgetAmount"],
                            ValidityDate=record["ValidityDate"],
                            Description=record["Description"],
                        ))
                        
                        inserted_records.append(record)
                        stats["inserted_records"] += 1

                except Exception as e:
                    stats["error_records"] += 1
                    stats["errors"].append(f"Error processing record: {str(e)}")

        db.add_all(new_record)

        await db.commit()

        # Generate result files
        inserted_df = pd.DataFrame(inserted_records) if inserted_records else None
        updated_df = pd.DataFrame(updated_records) if updated_records else None

        inserted_path = (
            save_df_to_temp_file(inserted_df)
            if inserted_df is not None and not inserted_df.empty
            else None
        )
        updated_path = (
            save_df_to_temp_file(updated_df)
            if updated_df is not None and not updated_df.empty
            else None
        )

        csv_id = str(uuid4())
        csv_store[csv_id] = {"inserted": inserted_path, "updated": updated_path}

        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "title": "Comparison Data Processed",
                "result": stats,
                "csv_id": csv_id,
                "operation_type": "comparison",
            },
        )

    except Exception as e:
        logger.error(f"Error processing comparison data: {str(e)}")
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "title": "Processing Error",
                "error": f"An error occurred while processing your file: {str(e)}",
                "result": {
                    "total_rows": 0,
                    "inserted_records": 0,
                    "updated_records": 0,
                    "error_records": 0,
                    "errors": [str(e)],
                },
                "csv_id": None,
                "operation_type": "comparison",
            },
        )


@router.get("/download/{csv_id}/{csv_type}")
def download_csv(csv_id: str, csv_type: str, background_tasks: BackgroundTasks):
    if csv_id not in csv_store:
        raise HTTPException(status_code=404, detail="CSV session not found")

    file_path = csv_store[csv_id].get(csv_type)
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"{csv_type} data not found")

    response = FileResponse(
        path=file_path,
        media_type="text/csv",
        filename=f"{csv_type}_records.csv",
    )

    # Schedule cleanup of the specific temp file only
    def cleanup_csv_file():
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Error deleting temp file {file_path}: {e}")
        csv_store[csv_id][csv_type] = None
        if all(value is None for value in csv_store[csv_id].values()):
            del csv_store[csv_id]

    background_tasks.add_task(cleanup_csv_file)
    return response
