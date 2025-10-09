import io
import csv
import tempfile
import logging
from typing import List, Dict, Any, Optional
from uuid import uuid4
import os

import pandas as pd
from fastapi import (
    APIRouter,
    File,
    UploadFile,
    HTTPException,
    Request,
    Query,
)
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from app.utils.excel_processor import read_excel_file, EXCEL_EXTENSIONS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/file-viewer", tags=["File Viewer"])
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))

# Store uploaded files in memory (in production, use proper storage)
uploaded_files: Dict[str, Dict[str, Any]] = {}


@router.get("/", response_class=HTMLResponse)
async def file_viewer_page(request: Request):
    """Main file viewer page"""
    return templates.TemplateResponse(
        "file_viewer.html", 
        {"request": request, "title": "CSV/Excel File Viewer"}
    )


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process CSV/Excel file"""
    try:
        # Validate file extension
        filename = file.filename.lower()
        file_ext = os.path.splitext(filename)[1]
        
        if file_ext not in ['.csv'] + EXCEL_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Allowed formats: CSV, XLSX, XLS"
            )
        
        # Generate unique file ID
        file_id = str(uuid4())
        
        # Process file based on extension
        if file_ext in EXCEL_EXTENSIONS:
            df = await read_excel_file(file)
        else:  # CSV
            content = await file.read()
            # Try different encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    csv_data = io.StringIO(content.decode(encoding))
                    df = pd.read_csv(csv_data)
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.warning(f"Failed to read CSV with {encoding}: {e}")
                    continue
            
            if df is None:
                raise HTTPException(
                    status_code=400, 
                    detail="Unable to decode CSV file. Please check file encoding."
                )
        
        # Validate DataFrame
        if df.empty:
            raise HTTPException(
                status_code=400, 
                detail="File contains no data"
            )
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Store file data
        uploaded_files[file_id] = {
            "filename": file.filename,
            "data": df,
            "columns": df.columns.tolist(),
            "row_count": len(df),
            "upload_time": pd.Timestamp.now().isoformat()
        }
        
        return JSONResponse({
            "file_id": file_id,
            "filename": file.filename,
            "columns": df.columns.tolist(),
            "row_count": len(df),
            "message": f"Successfully uploaded {file.filename} with {len(df)} rows and {len(df.columns)} columns"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing file: {str(e)}"
        )


@router.get("/data/{file_id}")
async def get_file_data(
    file_id: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=1000),
    search: Optional[str] = Query(None),
    sort_column: Optional[str] = Query(None),
    sort_order: str = Query("asc", regex="^(asc|desc)$")
):
    """Get paginated file data with filtering and sorting"""
    
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        df = uploaded_files[file_id]["data"].copy()
        
        # Apply search filter
        if search:
            search = search.lower()
            mask = False
            for column in df.columns:
                if df[column].dtype == 'object':
                    mask |= df[column].astype(str).str.lower().str.contains(search, na=False)
            df = df[mask]
        
        # Apply sorting
        if sort_column and sort_column in df.columns:
            df = df.sort_values(by=sort_column, ascending=(sort_order == "asc"))
        
        # Calculate pagination
        total_rows = len(df)
        total_pages = (total_rows + per_page - 1) // per_page
        start_idx = (page - 1) * per_page
        end_idx = min(start_idx + per_page, total_rows)
        
        # Get paginated data
        paginated_df = df.iloc[start_idx:end_idx]
        
        # Convert to records
        records = paginated_df.to_dict('records')
        
        # Handle NaN values
        for record in records:
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None
                elif isinstance(value, (pd.Timestamp)):
                    record[key] = value.isoformat()
        
        return {
            "data": records,
            "pagination": {
                "current_page": page,
                "total_pages": total_pages,
                "per_page": per_page,
                "total_rows": total_rows,
                "start_index": start_idx + 1,
                "end_index": end_idx
            },
            "columns": uploaded_files[file_id]["columns"],
            "filename": uploaded_files[file_id]["filename"]
        }
        
    except Exception as e:
        logger.error(f"Error retrieving data: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error retrieving data: {str(e)}"
        )


@router.get("/columns/{file_id}")
async def get_columns(file_id: str):
    """Get column information for a file"""
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        df = uploaded_files[file_id]["data"]
        columns_info = []
        
        for column in df.columns:
            col_info = {
                "name": column,
                "type": str(df[column].dtype),
                "unique_values": int(df[column].nunique()),
                "null_count": int(df[column].isnull().sum())
            }
            
            # Add sample values for categorical data
            if df[column].dtype == 'object' and df[column].nunique() <= 50:
                col_info["sample_values"] = df[column].dropna().unique().tolist()[:10]
            
            columns_info.append(col_info)
        
        return {
            "file_id": file_id,
            "columns": columns_info,
            "total_rows": len(df)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving columns: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error retrieving columns: {str(e)}"
        )


@router.post("/filter/{file_id}")
async def filter_data(
    file_id: str,
    filters: Dict[str, Any]
):
    """Apply column-specific filters to the data"""
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        df = uploaded_files[file_id]["data"].copy()
        
        # Apply filters
        for column, filter_config in filters.items():
            if column not in df.columns:
                continue
            
            filter_type = filter_config.get("type", "equals")
            value = filter_config.get("value")
            
            if value is None or value == "":
                continue
            
            if filter_type == "equals":
                df = df[df[column].astype(str) == str(value)]
            elif filter_type == "contains":
                df = df[df[column].astype(str).str.contains(str(value), case=False, na=False)]
            elif filter_type == "starts_with":
                df = df[df[column].astype(str).str.startswith(str(value), na=False)]
            elif filter_type == "ends_with":
                df = df[df[column].astype(str).str.endswith(str(value), na=False)]
            elif filter_type == "greater_than":
                try:
                    numeric_value = float(value)
                    df = df[pd.to_numeric(df[column], errors='coerce') > numeric_value]
                except:
                    pass
            elif filter_type == "less_than":
                try:
                    numeric_value = float(value)
                    df = df[pd.to_numeric(df[column], errors='coerce') < numeric_value]
                except:
                    pass
            elif filter_type == "not_empty":
                df = df[df[column].notna() & (df[column] != "")]
            elif filter_type == "is_empty":
                df = df[df[column].isna() | (df[column] == "")]
        
        # Store filtered data temporarily
        filtered_file_id = f"filtered_{file_id}_{uuid4()}"
        uploaded_files[filtered_file_id] = {
            "filename": f"filtered_{uploaded_files[file_id]['filename']}",
            "data": df,
            "columns": df.columns.tolist(),
            "row_count": len(df),
            "is_filtered": True,
            "original_file_id": file_id
        }
        
        return JSONResponse({
            "filtered_file_id": filtered_file_id,
            "row_count": len(df),
            "columns": df.columns.tolist(),
            "message": f"Applied filters. {len(df)} rows match the criteria."
        })
        
    except Exception as e:
        logger.error(f"Error applying filters: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error applying filters: {str(e)}"
        )


@router.get("/download/{file_id}")
async def download_file(file_id: str, format: str = Query("csv", regex="^(csv|xlsx)$")):
    """Download file in CSV or Excel format"""
    
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        df = uploaded_files[file_id]["data"]
        filename = uploaded_files[file_id]["filename"]
        
        # Generate download filename
        base_name = os.path.splitext(filename)[0]
        if uploaded_files[file_id].get("is_filtered"):
            base_name = f"filtered_{base_name}"
        
        if format == "csv":
            # Create CSV
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()
            
            return StreamingResponse(
                io.BytesIO(csv_content.encode('utf-8-sig')),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename={base_name}.csv",
                    "Content-Type": "text/csv; charset=utf-8"
                }
            )
        
        else:  # Excel
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
            
            excel_buffer.seek(0)
            
            return StreamingResponse(
                excel_buffer,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={
                    "Content-Disposition": f"attachment; filename={base_name}.xlsx"
                }
            )
        
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error downloading file: {str(e)}"
        )


@router.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """Delete uploaded file"""
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        del uploaded_files[file_id]
        return JSONResponse({"message": "File deleted successfully"})
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error deleting file: {str(e)}"
        )