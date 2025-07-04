from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import redis
from typing import Optional
import time
from app.core.config import settings
from app.core.logging import logger
import hashlib
import secrets


# Rate limiter setup
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{settings.RATE_LIMIT_PER_MINUTE}/minute"]
)


class RateLimitManager:
    """Custom rate limiting with Redis backend."""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        
    def _get_key(self, identifier: str, endpoint: str) -> str:
        """Generate rate limit key."""
        return f"rate_limit:{endpoint}:{identifier}"
    
    async def is_rate_limited(self, identifier: str, endpoint: str, limit: int, window: int) -> bool:
        """Check if request is rate limited."""
        if not self.redis_client:
            return False
        
        try:
            key = self._get_key(identifier, endpoint)
            current_time = int(time.time())
            
            # Remove old entries outside the window
            self.redis_client.zremrangebyscore(key, 0, current_time - window)
            
            # Count current requests
            current_count = self.redis_client.zcard(key)
            
            if current_count >= limit:
                return True
            
            # Add current request
            self.redis_client.zadd(key, {str(current_time): current_time})
            self.redis_client.expire(key, window)
            
            return False
            
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            return False  # Fail open
    
    async def get_rate_limit_info(self, identifier: str, endpoint: str, window: int) -> dict:
        """Get rate limit information for identifier."""
        if not self.redis_client:
            return {"requests": 0, "window": window, "remaining": "unknown"}
        
        try:
            key = self._get_key(identifier, endpoint)
            current_time = int(time.time())
            
            # Remove old entries
            self.redis_client.zremrangebyscore(key, 0, current_time - window)
            
            # Count current requests
            current_count = self.redis_client.zcard(key)
            
            return {
                "requests": current_count,
                "window": window,
                "reset_time": current_time + window
            }
            
        except Exception as e:
            logger.error(f"Rate limit info error: {e}")
            return {"requests": 0, "window": window, "remaining": "error"}


class SecurityManager:
    """Security utilities and middleware."""
    
    def __init__(self):
        self.rate_limiter = RateLimitManager()
    
    def generate_api_key(self) -> str:
        """Generate a secure API key."""
        return secrets.token_urlsafe(32)
    
    def hash_api_key(self, api_key: str) -> str:
        """Hash an API key for storage."""
        return hashlib.sha256(f"{api_key}{settings.SECRET_KEY}".encode()).hexdigest()
    
    def validate_api_key(self, provided_key: str, stored_hash: str) -> bool:
        """Validate an API key against stored hash."""
        return self.hash_api_key(provided_key) == stored_hash
    
    def get_client_identifier(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Try to get real IP from headers (for reverse proxy setups)
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        return request.client.host if request.client else "unknown"
    
    def is_suspicious_request(self, request: Request) -> bool:
        """Check if request appears suspicious."""
        user_agent = request.headers.get("User-Agent", "").lower()
        
        # Block common bot patterns
        suspicious_patterns = [
            "bot", "crawler", "spider", "scraper", "curl", "wget",
            "python-requests", "automated"
        ]
        
        for pattern in suspicious_patterns:
            if pattern in user_agent:
                return True
        
        return False
    
    def sanitize_query(self, query: str) -> str:
        """Sanitize search query to prevent injection attacks."""
        # Remove potentially dangerous characters
        dangerous_chars = ["<", ">", "&", "\"", "'", ";", "(", ")", "{", "}", "[", "]"]
        sanitized = query
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, "")
        
        # Limit length
        sanitized = sanitized[:200]
        
        # Remove extra whitespace
        sanitized = " ".join(sanitized.split())
        
        return sanitized


# Security middleware
async def security_middleware(request: Request, call_next):
    """Security middleware for all requests."""
    security_manager = SecurityManager()
    
    # Check for suspicious requests
    if security_manager.is_suspicious_request(request):
        logger.warning(f"Suspicious request from {security_manager.get_client_identifier(request)}: {request.headers.get('User-Agent')}")
        # For now, just log it - could block in production
    
    # Add security headers
    response = await call_next(request)
    
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    if not settings.DEBUG:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    return response


# API Key authentication (optional)
class APIKeyBearer(HTTPBearer):
    """API Key authentication scheme."""
    
    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)
        self.security_manager = SecurityManager()
    
    async def __call__(self, request: Request) -> Optional[HTTPAuthorizationCredentials]:
        credentials = await super().__call__(request)
        
        if credentials:
            # In a real implementation, you'd validate against a database
            # For now, just check if it's a valid format
            if len(credentials.credentials) < 16:
                raise HTTPException(
                    status_code=403,
                    detail="Invalid API key format"
                )
        
        return credentials


# Rate limiting decorators
def rate_limit_endpoint(requests_per_minute: int = None):
    """Decorator for endpoint-specific rate limiting."""
    def decorator(func):
        func._rate_limit = requests_per_minute or settings.RATE_LIMIT_PER_MINUTE
        return func
    return decorator


# CORS security
def get_cors_origins():
    """Get CORS origins based on environment."""
    if settings.DEBUG:
        return ["*"]  # Allow all origins in development
    else:
        # In production, specify exact origins
        return [
            "https://your-frontend-domain.com",
            "https://www.your-frontend-domain.com"
        ]


# Content Security Policy
def get_csp_header():
    """Get Content Security Policy header."""
    if settings.DEBUG:
        return "default-src 'self' 'unsafe-inline' 'unsafe-eval'; connect-src 'self' ws: wss:"
    else:
        return "default-src 'self'; connect-src 'self'"


# Initialize security components
security_manager = SecurityManager()
api_key_bearer = APIKeyBearer(auto_error=False)


# Helper functions for endpoints
async def check_rate_limit(request: Request, endpoint: str, limit: int = None):
    """Check rate limit for specific endpoint."""
    if not limit:
        limit = settings.RATE_LIMIT_PER_MINUTE
    
    client_id = security_manager.get_client_identifier(request)
    
    if await security_manager.rate_limiter.is_rate_limited(client_id, endpoint, limit, 60):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": "60"}
        )


def sanitize_user_input(text: str) -> str:
    """Sanitize user input text."""
    return security_manager.sanitize_query(text)