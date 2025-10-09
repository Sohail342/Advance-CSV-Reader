from sqlalchemy import Column, String, Index, Integer
from app.utils.database import Base


class CostCenter(Base):
    """SQLAlchemy model for Cost Centers with required fields"""

    __tablename__ = "cost_centers"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    BranchCode = Column(String, nullable=False, index=True)
    CostCenterID = Column(String, nullable=True)
    CostCenterName = Column(String, nullable=True)
    Region = Column(String, nullable=True)
    BranchName = Column(String, nullable=True)
    CostCenterDescription = Column(String, nullable=True)

    # Create indexes for performance optimization
    __table_args__ = (
        Index("idx_branchcode", BranchCode),
    )

    def __repr__(self):
        return f"<CostCenter(cost_center_id='{self.CostCenterID}', cost_center_name='{self.CostCenterName}')>"