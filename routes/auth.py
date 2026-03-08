"""
Auth routes — Firebase token verification middleware
"""
from fastapi import APIRouter, HTTPException, Header
from services.firebase import get_auth

router = APIRouter()


@router.post("/verify")
def verify_token(authorization: str = Header(...)):
    """
    Verify a Firebase ID token from the React frontend.
    The frontend should send: Authorization: Bearer <firebase_id_token>
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token = authorization.split("Bearer ")[1]
    try:
        firebase_auth = get_auth()
        decoded       = firebase_auth.verify_id_token(token)
        return {
            "uid":   decoded["uid"],
            "email": decoded.get("email", ""),
            "valid": True,
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")


@router.get("/me")
def get_current_user(authorization: str = Header(...)):
    """Get current user info from Firebase token."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.split("Bearer ")[1]
    try:
        firebase_auth = get_auth()
        decoded       = firebase_auth.verify_id_token(token)
        user          = firebase_auth.get_user(decoded["uid"])
        return {
            "uid":         user.uid,
            "email":       user.email,
            "displayName": user.display_name,
            "photoURL":    user.photo_url,
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))
