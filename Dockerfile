# AutoClip Dockerfile
# 多阶段构建，优化镜像大小

# 第一阶段：构建前端
FROM node:18-bookworm-slim AS frontend-builder

WORKDIR /app/frontend

# 安装必要的系统依赖
# 复制前端依赖文件
COPY frontend/package*.json ./

# 安装前端依赖（使用完整安装，包括devDependencies）
RUN npm config set registry https://registry.npmmirror.com \
    && npm config set fetch-retries 5 \
    && npm ci

# 复制前端源代码
COPY frontend/ ./

# 构建前端
RUN npm run build

# 第二阶段：构建后端
FROM python:3.9-slim-bookworm AS backend-builder

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 安装系统依赖
# 复制Python依赖文件
COPY requirements.txt ./

# 安装Python依赖
RUN pip install --no-cache-dir --retries 8 --timeout 120 \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt

# 第三阶段：最终镜像
FROM python:3.9-slim-bookworm

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# 创建非root用户
RUN groupadd -r autoclip && useradd -r -g autoclip autoclip

# 安装运行时依赖
RUN sed -i \
      -e 's|http://deb.debian.org/debian|https://mirrors.aliyun.com/debian|g' \
      -e 's|http://deb.debian.org/debian-security|https://mirrors.aliyun.com/debian-security|g' \
      /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
    ffmpeg \
    curl \
    fonts-noto-cjk \
    fontconfig \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 设置工作目录
WORKDIR /app

# 从构建阶段复制文件
COPY --from=backend-builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=backend-builder /usr/local/bin /usr/local/bin
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# 复制项目文件
COPY backend/ ./backend/
COPY scripts/ ./scripts/
COPY *.sh ./
COPY env.example .env
COPY docker-entrypoint.sh ./
RUN sed -i 's/\r$//' docker-entrypoint.sh *.sh

# 创建必要的目录
RUN mkdir -p data/projects data/uploads data/temp data/output logs

# 设置权限
RUN chown -R autoclip:autoclip /app
RUN chmod +x *.sh
RUN chmod +x docker-entrypoint.sh
RUN chmod -R 755 data logs

# 切换到非root用户
USER autoclip

# 暴露端口
EXPOSE 8000 3000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health/ || exit 1

# 启动命令
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
