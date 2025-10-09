import pandas as pd
import io
import logging
from typing import List, Dict, Any
from fastapi import UploadFile

# Configure logging
logger = logging.getLogger(__name__)

# Supported Excel file extensions
EXCEL_EXTENSIONS = [".xlsx", ".xls"]


async def read_excel_file(file: UploadFile) -> pd.DataFrame:
    """
    Read Excel file and return a pandas DataFrame
    """
    try:
        # Read file content
        content = await file.read()

        # Create BytesIO object from content
        excel_data = io.BytesIO(content)

        # Determine Excel engine based on file extension
        if file.filename.endswith(".xlsx"):
            engine = "openpyxl"
        else:  # .xls
            engine = "xlrd"

        # Read Excel file into DataFrame
        df = pd.read_excel(excel_data, engine=engine)

        # Reset file pointer for potential future reads
        await file.seek(0)

        return df

    except Exception as e:
        logger.error(f"Error reading Excel file: {str(e)}")
        raise ValueError(f"Error reading Excel file: {str(e)}")


def validate_excel_headers(df: pd.DataFrame, required_headers: List[str]) -> None:
    """
    Validate that the Excel file contains the required headers
    """
    # Convert DataFrame columns to lowercase for case-insensitive comparison
    df_headers = [col.strip().lower() for col in df.columns]

    # Check for missing headers
    missing_headers = [h for h in required_headers if h.lower() not in df_headers]
    if missing_headers:
        raise ValueError(f"Missing required headers: {', '.join(missing_headers)}")


def excel_to_records(
    df: pd.DataFrame, required_headers: List[str]
) -> List[Dict[str, Any]]:
    """
    Convert Excel DataFrame to a list of records
    """
    # Convert DataFrame columns to lowercase for case-insensitive comparison
    df.columns = [col.strip() for col in df.columns]

    # Convert DataFrame to list of dictionaries
    records = df.to_dict(orient="records")

    # Validate and clean records
    cleaned_records = []
    for i, record in enumerate(
        records, start=2
    ):  # Start from 2 to account for header row
        # Skip empty rows
        if all(pd.isna(v) or v == "" for v in record.values()):
            continue

        # Clean record
        cleaned_record = {}
        for header in required_headers:
            # Get value, handle NaN
            value = record.get(header, "")
            if pd.isna(value):
                value = ""
            elif not isinstance(value, str):
                value = str(value).strip()
            else:
                value = value.strip()

            cleaned_record[header] = value

        # Validate primary key
        # if not cleaned_record["SubGLCode"]:
        #     logger.warning(f"Row {i}: Missing primary key (SubGLCode)")
        #     continue

        cleaned_records.append(cleaned_record)

    return cleaned_records
