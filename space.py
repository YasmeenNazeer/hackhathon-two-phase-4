# space.py - Main entry point for Hugging Face Space
import os
import uvicorn
from app import app

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 7860)),
        log_level="info"
    )