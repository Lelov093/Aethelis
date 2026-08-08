from __future__ import annotations


class ProductApplicationError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ProductNotFoundError(ProductApplicationError):
    pass


class ProductAccessDeniedError(ProductApplicationError):
    pass


class ProductConflictError(ProductApplicationError):
    pass


class ProductCompatibilityError(ProductApplicationError):
    pass
