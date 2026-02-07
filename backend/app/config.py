from pydantic_settings import BaseSettings, SettingsConfigDict

import os

class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./elevate_app.db"  # Default to SQLite
    JWT_SECRET: str = "secure-jwt-secret-for-demonstration"
    ALGORITHM: str = "HS256"
    GEMINI_API_KEY: str = "your-gemini-api-key-here" # Added Gemini API Key

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    )

settings = Settings()
