# app/scripts/test_arq_connection.py
import asyncio
from arq import create_pool
from arq.connections import RedisSettings
from app.core.config import settings

async def main():
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    pool = await create_pool(redis_settings)
    await pool.set("arq_test_key", "hello")
    value = await pool.get("arq_test_key")
    print(f"arq connected successfully. Test value: {value}")
    await pool.aclose()

if __name__ == "__main__":
    asyncio.run(main())