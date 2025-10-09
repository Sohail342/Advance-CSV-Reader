from app.utils.database import Base
from sqlalchemy import Column, String, Index, Integer

class SubGLCodes(Base):
    """SQLAlchemy model for Sub GL Codes with required fields"""

    __tablename__ = "sub_gl_codes"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Required fields
    SubGLCode = Column(String, nullable=False, index=True)
    SubHeadID = Column(String, nullable=True)
    SubHeadName = Column(String, nullable=True)
    Head = Column(String, nullable=True)
    HeadID = Column(String, nullable=True)

    # Create indexes for performance optimization
    __table_args__ = (  
        Index("idx_SubGLCode", SubGLCode),
    )

    def __repr__(self):
        return f"<SubGLCodes(sub_gl_code='{self.SubGLCode}', region='{self.Region}')>"