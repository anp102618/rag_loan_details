from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from jose import jwt, JWTError
from fastapi.security import OAuth2PasswordBearer
from fastapi.security import OAuth2PasswordRequestForm


from src.db.main import get_session
from src.Users.models import User
from src.Users.schemas import UserCreate, UserOut, Token
from src.Users.auth import authenticate_user, get_current_user
from src.Users.helpers import (
    hash_password,
    create_access_token,
    create_refresh_token,
    SECRET_KEY,
    ALGORITHM
)

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# simple in-memory blacklist (replace with Redis in production)
blacklist = set()


# =========================
# REGISTER
# =========================
@router.post("/register", response_model=UserOut)
async def register(user: UserCreate, db: AsyncSession = Depends(get_session)):
    result = await db.execute(
        select(User).where(User.email == user.email)
    )
    existing = result.scalar_one_or_none()

    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    new_user = User(
        email=user.email,
        hashed_password=hash_password(user.password)
    )

    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    return new_user


# =========================
# LOGIN
# =========================

@router.post("/login")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_session)
):
    user = await authenticate_user(
        db,
        form_data.username,   # Swagger sends "username"
        form_data.password
    )

    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token(
        {"user_id": user.user_id}
    )

    return {
        "access_token": access_token,
        "token_type": "bearer"
    }


# =========================
# REFRESH TOKEN
# =========================
@router.post("/refresh", response_model=Token)
async def refresh_token(refresh_token: str):
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])

        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")

        user_id = payload.get("user_id")

        return {
            "access_token": create_access_token({"user_id": user_id}),
            "refresh_token": create_refresh_token({"user_id": user_id}),
        }

    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")


# =========================
# CURRENT USER
# =========================
@router.get("/me", response_model=UserOut)
async def get_me(current_user=Depends(get_current_user)):
    return current_user


# =========================
# LOGOUT
# =========================
@router.post("/logout")
async def logout(token: str = Depends(oauth2_scheme)):
    blacklist.add(token)
    return {"message": "Logged out"}