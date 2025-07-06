# 使用Python 3.11作为基础镜像
FROM python:3.11-slim

LABEL name="browser-use-api"

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    DEFAULT_AI_PROVIDER=google \
    BROWSER_USE_HEADFUL=false \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# 安装系统依赖项（playwright需要的基础依赖）
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    ca-certificates \
    fonts-liberation \
    libatk-bridge2.0-0 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libxss1 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# 先复制requirements.txt并安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 创建playwright浏览器目录
RUN mkdir -p /ms-playwright

# 安装playwright浏览器（需要在创建用户前安装）
RUN playwright install --with-deps chromium

# 创建非root用户并设置家目录
RUN groupadd -r appuser && useradd -r -g appuser -m appuser

# 复制项目文件（排除media和data文件夹）
COPY . .

# 创建必要的目录
RUN mkdir -p /app/media /app/data /app/task_storage

# 创建用户配置目录
RUN mkdir -p /home/appuser/.config /home/appuser/.local /home/appuser/.cache

# 创建browser-use特定目录
RUN mkdir -p /home/appuser/.config/browseruse/profiles/default

# 设置文件所有者
RUN chown -R appuser:appuser /app /home/appuser /ms-playwright

# 切换到非root用户
USER appuser

# 暴露端口
EXPOSE 8000

# 设置启动命令
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
