from fastapi import Depends, Security, HTTPException, status
from fastapi.security import APIKeyHeader
from db.session import get_db
from gateway.auth import verify_hash
from sqlalchemy.ext.asyncio import  AsyncSession


api_key_header = APIKeyHeader(name="X-API-KEY")
async def verify_api_key(api_key: str = Security(api_key_header), db: AsyncSession = Depends(get_db)):

    (
        api_key_id,
        api_key_is_active, 
        api_key_tenant_id, 
        tenant_record 
    ) = await verify_hash(db, api_key)

    if api_key_id is None or tenant_record is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or Revoked API key"
        )

    if not tenant_record.is_active or  not api_key_is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account currently not active"
            )

    return (
        api_key_id,
        api_key_is_active, 
        api_key_tenant_id, 
        tenant_record 
    )


