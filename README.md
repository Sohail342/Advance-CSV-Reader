# Data Processor

A FastAPI application that processes CSV and Excel data and synchronizes it with an SQLAlchemy database.

## Features

- FastAPI endpoint integrated with Jinja2 templating
- SQLAlchemy model for data headers
- CSV and Excel file upload and validation
- Support for multiple file formats (.csv, .xlsx, .xls)
- Database synchronization with proper indexing
- Comprehensive error handling
- Performance optimizations (bulk inserts, connection pooling)
- Detailed processing statistics

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

The application uses environment variables for configuration. You can create a `.env` file in the root directory with the following variables:

```
DATABASE_URL=sqlite:///./csv_data.db
```

For PostgreSQL:

```
DATABASE_URL=postgresql://user:password@localhost/dbname
```

## Running the Application

```bash
python run.py
```

The application will be available at http://localhost:8000

## File Format

The CSV or Excel file must contain the following headers in the first row:

- `sub_gl_code` (Primary key)
- `sub_head`
- `region`
- `branch_code`
- `branch_name`

### Supported File Formats

- CSV (.csv)
- Excel (.xlsx, .xls)

## API Endpoints

- `GET /` - Home page
- `GET /csv/upload` - File upload form
- `POST /csv/upload` - Process data file and display results
- `POST /csv/api/upload` - API endpoint for data file processing
- `GET /health` - Health check endpoint

## Database Model

The application uses the `CSVHeaders` model with the following fields:

- `sub_gl_code` (String, primary key)
- `sub_head` (String)
- `region` (String)
- `branch_code` (String)
- `branch_name` (String)

## Performance Optimizations

- Bulk insert operations using SQLAlchemy's bulk_save_objects
- Database indexes on matching fields (sub_gl_code, branch_code)
- Asynchronous I/O operations for file processing
- Connection pooling for database access