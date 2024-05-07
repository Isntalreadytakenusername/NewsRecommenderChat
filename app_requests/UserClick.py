from pydantic import BaseModel

class UserClick(BaseModel):
    user_id: str
    title: str
    date: str
    domain: str