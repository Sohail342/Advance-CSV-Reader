from sqlalchemy import Column, String, Index, Integer
from app.utils.database import Base


class CSVHeaders(Base):
    """SQLAlchemy model for CSV headers with required fields"""

    __tablename__ = "csv_headers"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Required fields
    sub_gl_code = Column(String, nullable=False, index=True)
    sub_head = Column(String, nullable=False)
    region = Column(String, nullable=False)
    branch_code = Column(String, nullable=False, index=True)
    branch_name = Column(String, nullable=False)

    # Create indexes for performance optimization
    __table_args__ = (
        Index("idx_branch_code", branch_code),
        Index("idx_region", region),
        # Composite index for common query patterns
        Index("idx_branch_region", branch_code, region),
    )

    def __repr__(self):
        return f"<CSVHeaders(sub_gl_code='{self.sub_gl_code}', branch_code='{self.branch_code}', region='{self.region}')>"
