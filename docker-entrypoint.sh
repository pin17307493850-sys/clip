#!/bin/bash
set -euo pipefail

export PYTHONPATH=/app
export PYTHONUNBUFFERED=1

mkdir -p /app/data/projects /app/data/uploads /app/data/temp /app/data/output /app/logs

if [[ ! -f /app/data/autoclip.db ]]; then
    echo "初始化数据库..."
    python -c "
import sys
sys.path.insert(0, '/app')
from backend.core.database import engine, Base
from backend.models import project, task, clip, collection, bilibili
Base.metadata.create_all(bind=engine)
print('数据库初始化成功')
"
fi

echo "检查Redis连接..."
python -c "
import os
import redis
try:
    redis_url = os.getenv('REDIS_URL', 'redis://redis:6379/0')
    redis.Redis.from_url(redis_url, decode_responses=True).ping()
    print(f'Redis连接成功: {redis_url}')
except Exception as e:
    print(f'Redis连接失败: {e}')
"

echo "启动AutoClip应用..."
exec "$@"
