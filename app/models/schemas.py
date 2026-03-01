from pydantic import BaseModel
from typing import List, Dict

class TextRequest(BaseModel):
    text: str

class EntityResponse(BaseModel):
    entities: List[Dict]