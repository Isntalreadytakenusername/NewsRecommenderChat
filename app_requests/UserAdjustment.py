from pydantic import BaseModel

# model app POST request to save user click
class UserAdjustment(BaseModel):
    user_id: str
    request: str