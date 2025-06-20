from fastapi import HTTPException
from .error_code import ErrorCode

class AppException(HTTPException):
    def __init__(self, status_code: int, code: ErrorCode, message: str):
        super().__init__(
            status_code=status_code,
            detail={"code": code, "message": message}
        )
        self.code = code
        self.message = message
