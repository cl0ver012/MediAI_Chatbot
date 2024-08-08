from pydantic import Field, BaseModel
from typing import List

class Message(BaseModel):
    role: str
    content: str
