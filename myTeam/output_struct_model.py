from pydantic import BaseModel

class OutputStruct(BaseModel):
    """A structured output for testing purposes."""
    city: str
    description: str
