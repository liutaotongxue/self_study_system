"""全局配置,从 .env 读取。"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    google_api_key: str = ""               # S4 vision OCR (Gemini),Anthropic filter 规避路径
    database_url: str = "sqlite:///./app.db"
    env: str = "dev"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
