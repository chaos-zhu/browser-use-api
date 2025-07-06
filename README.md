# Browser Use API

> 更新了下henry0hai/browser-n8n-local，发现有些日子没维护跑不起来了，于是优化(反向)了下.

> 原项目地址: https://github.com/henry0hai/browser-n8n-local

基于browser-use构建的api，只需要docker-compose即可一键启动

## docker-compose.yml启动
```bash
mkdir browser-use-api
cd browser-use-api

wget https://raw.githubusercontent.com/chaos-zhu/browser-use-api/refs/heads/main/.env-example

wget https://raw.githubusercontent.com/chaos-zhu/browser-use-api/refs/heads/main/docker-compose.yml

mv .env-example .env

# 填写谷歌gemini Key到 .env中的GOOGLE_API_KEY

docker-compose up -d
```

## 本地构建

```bash
docker build -t chaoszhu/browser-use-api:latest .
```

## 本地启动

```bash
python -m venv venv

venv\Scripts\activate
# On Mac: source venv/bin/activate

pip install -r requirements.txt

playwright install # --with-deps chromium

cp .env-example .env

python app.py

```
The server will start at http://localhost:8000 by default.

You can access the API documentation at http://localhost:8000/docs

## API

文档地址： http://{your_ip}:8000/docs

| Method | Endpoint                           | Description                  |
|--------|------------------------------------|------------------------------|
| POST   | /api/v1/run-task                   | Start a new browser task     |
| GET    | /api/v1/task/{task_id}             | Get task details             |
| GET    | /api/v1/task/{task_id}/status      | Get task status              |
| PUT    | /api/v1/stop-task/{task_id}        | Stop a running task          |
| PUT    | /api/v1/pause-task/{task_id}       | Pause a running task         |
| PUT    | /api/v1/resume-task/{task_id}      | Resume a paused task         |
| GET    | /api/v1/list-tasks                 | List all tasks               |
| GET    | /live/{task_id}                    | Live view UI                 |
| GET    | /api/v1/ping                       | Check health                 |
| GET    | /api/v1/task/{task_id}/media       | Get task media               |
| GET    | /api/v1/task/{task_id}/media/list  | List all media from task     |
| GET    | /api/v1/media/{task_id}/{filename} | Display task media content   |

## 示例

### 开始任务

```bash
curl -X POST http://localhost:8000/api/v1/run-task \
  -H "Content-Type: application/json" \
  -d '{"task": "Go to google.com and search for chaoszhu github"}'
```

## 感谢

- [Browser Use](https://github.com/browser-use/browser-use) - The underlying browser automation library
- [FastAPI](https://fastapi.tiangolo.com/) - The web framework used
- [n8n](https://n8n.io/) - The workflow automation platform this bridge is designed for # browser-n8n-local
