from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional


class Settings(BaseSettings):
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-3-flash-preview", alias="GEMINI_MODEL")
    extraction_max_retries: int = Field(default=3, alias="EXTRACTION_MAX_RETRIES", ge=1, le=5)
    max_input_characters: int = Field(default=120000, alias="MAX_INPUT_CHARACTERS", ge=1000)
    chunk_size_characters: int = Field(default=8000, alias="CHUNK_SIZE_CHARACTERS", ge=1000)
    chunk_overlap_characters: int = Field(default=500, alias="CHUNK_OVERLAP_CHARACTERS", ge=0)
    max_chunks: int = Field(default=20, alias="MAX_CHUNKS", ge=1)
    required_fields: str = Field(default="buyer.name,seller.name,agreement.date", alias="REQUIRED_FIELDS")
    enable_ocr_fallback: bool = Field(default=True, alias="ENABLE_OCR_FALLBACK")
    review_db_path: str = Field(default="review_sessions.db", alias="REVIEW_DB_PATH")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def required_field_paths(self) -> List[str]:
        return [field.strip() for field in self.required_fields.split(",") if field.strip()]
