import jwt
from datetime import datetime, timedelta
from typing import Optional
from passlib.context import CryptContext
from app.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthService:
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify plain password against hashed password."""
        # For simplicity and ease of config, we can also check plain comparison if the password is plain
        # but let's support both bcrypt and fallback plain comparison.
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception:
            # Fallback to direct comparison if hashing fails or is not formatted
            return plain_password == hashed_password

    def get_password_hash(self, password: str) -> str:
        """Hash a password using bcrypt."""
        return pwd_context.hash(password)

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Generate JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=8)
            
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET, algorithm="HS256")
        return encoded_jwt

    def verify_token(self, token: str) -> Optional[dict]:
        """Verify JWT token and return payload."""
        try:
            payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
            return payload
        except jwt.PyJWTError:
            return None

# Global singleton instance
auth_service = AuthService()
