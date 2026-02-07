import re
from fastapi import HTTPException, Header
from .config import settings

def validate_user_id(user_id: str) -> bool:
    """Validate user ID format"""
    if not user_id or not isinstance(user_id, str):
        return False
    # Check that user ID is alphanumeric with hyphens/underscores, length 1-64
    if not re.match(r'^[a-zA-Z0-9_-]{1,64}$', user_id):
        return False
    return True

async def get_current_user_id(x_user_id: str = Header(None)):
    user_id = x_user_id

    if not user_id:
        raise HTTPException(status_code=401, detail="Missing X-User-ID header")

    if not validate_user_id(user_id):
        raise HTTPException(status_code=401, detail="Invalid X-User-ID format")

    return user_id