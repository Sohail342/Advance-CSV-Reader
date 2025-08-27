from pydantic import BaseModel


class CSVRecordOut(BaseModel):
    sub_gl_code: str
    branch_code: str
    sub_head: str
    region: str
    branch_name: str

    class Config:
        orm_mode = True
