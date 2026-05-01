from pydantic import BaseModel, Field, StrictStr, field_validator


class ChatRequest(BaseModel):
    message: StrictStr = Field(default="", max_length=2000)
    session_id: StrictStr = Field(default="default", max_length=100)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        session_id = value.strip()
        if not session_id:
            raise ValueError("session_id must not be empty")
        return session_id
