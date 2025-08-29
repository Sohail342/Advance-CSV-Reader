from sqlalchemy import Column, String, Index, Integer
from app.utils.database import Base


class CSVHeaders(Base):
    """SQLAlchemy model for CSV headers with required fields"""

    __tablename__ = "csv_headers"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Required fields
    CostCenterID = Column(Integer, nullable=False, index=True)
    SubHeadID = Column(Integer, nullable=True)
    SubGLCode = Column(String, nullable=False, index=True)
    SubHead = Column(String, nullable=True)
    Region = Column(String, nullable=True)
    BCode = Column(String, nullable=True)
    BName = Column(String, nullable=True)
    BudgetID = Column(Integer, nullable=True)
    Head = Column(String, nullable=True)
    HeadID	= Column(Integer, nullable=True)
    CostCenter = Column(String, nullable=True)
    BudgetAmount = Column(String, nullable=True)
    ValidityDate = Column(String, nullable=True)
    Description = Column(String, nullable=True)


    # Create indexes for performance optimization
    __table_args__ = (
        Index("idx_CostCenterID", CostCenterID),

        # Composite index for common query patterns
        Index("idx_CostCenterID_SubGLCode", CostCenterID, SubGLCode),
    )

    def __repr__(self):
        return f"<CSVHeaders(sub_gl_code='{self.SubGLCode}', CostCenterID='{self.CostCenterID}', region='{self.Region}')>"
