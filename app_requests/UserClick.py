from pydantic import BaseModel

# model app POST request to save user click
class UserClick(BaseModel):
    user_id: str
    title: str
    date: str
    domain: str