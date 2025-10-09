from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from app.routes import csv_routes, insert_update_routes, sub_glcode, cost_center, file_viewer_routes


# Create FastAPI app
app = FastAPI(
    title="CSV Data Processor",
    description="FastAPI application for processing CSV data and synchronizing with database",
    version="1.0.0",
)

# Setup templates
templates_path = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_path))


# Include routers
app.include_router(csv_routes.router)
app.include_router(insert_update_routes.router)
app.include_router(sub_glcode.router)
app.include_router(cost_center.router)
app.include_router(file_viewer_routes.router)


# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "title": "CSV Data Processor"}
    )

@app.get("/file-viewer")
async def file_viewer_page(request: Request):
    return templates.TemplateResponse("file_viewer.html", {"request": request})


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
