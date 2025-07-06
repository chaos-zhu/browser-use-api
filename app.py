from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
import base64
import mimetypes

from typing import Optional
from datetime import datetime, UTC
from enum import Enum

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response, Query, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from browser_use.llm import (
    ChatOpenAI,
    ChatAnthropic,
    ChatGoogle,
    ChatOllama,
    ChatGroq,
    ChatAWSBedrock,
    ChatAzureOpenAI
)
from pydantic import BaseModel

# This import will work once browser-use is installed
# For development, you may need to add the browser-use repo to your PYTHONPATH
from browser_use import Agent
from browser_use.agent.views import AgentHistoryList
from browser_use import BrowserConfig, Browser
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext

from pathlib import Path

# Import our task storage abstraction
from task_storage import get_task_storage
from task_storage.base import DEFAULT_USER_ID


# Define task status enum
class TaskStatus(str, Enum):
    CREATED = "created"  # Task is initialized but not yet started
    RUNNING = "running"  # Task is currently executing
    FINISHED = "finished"  # Task has completed successfully
    STOPPED = "stopped"  # Task was manually stopped
    PAUSED = "paused"  # Task execution is temporarily paused
    FAILED = "failed"  # Task encountered an error and could not complete
    STOPPING = "stopping"  # Task is in the process of stopping (transitional state)


# Load environment variables from .env file
load_dotenv()

# Create media directory if it doesn't exist
MEDIA_DIR = Path("media")
MEDIA_DIR.mkdir(exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("browser-use-bridge")

app = FastAPI(title="Browser Use Bridge API")

# Mount static files
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")


# 用于枚举序列化的自定义JSON编码器
class EnumJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


# 配置FastAPI使用自定义JSON序列化响应
@app.middleware("http")
async def add_json_serialization(request: Request, call_next):
    response = await call_next(request)

    # 仅尝试修改JSON响应并检查body()方法是否存在
    if response.headers.get("content-type") == "application/json" and hasattr(
        response, "body"
    ):
        try:
            content = await response.body()
            content_str = content.decode("utf-8")
            content_dict = json.loads(content_str)
            # 将任何枚举值转换为其字符串表示形式
            content_str = json.dumps(content_dict, cls=EnumJSONEncoder)
            response = Response(
                content=content_str,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type="application/json",
            )
        except Exception as e:
            logger.error(f"序列化JSON响应时出错: {str(e)}")

    return response


# 启用CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化任务存储
task_storage = get_task_storage()


# 模型
class TaskRequest(BaseModel):
    task: str
    ai_provider: Optional[str] = None  # 如果未提供，将设置为DEFAULT_AI_PROVIDER环境变量
    save_browser_data: Optional[bool] = False  # 是否保存浏览器cookie
    headful: Optional[bool] = None  # 覆盖BROWSER_USE_HEADFUL设置
    use_custom_chrome: Optional[bool] = (
        None  # 是否使用来自环境变量的自定义Chrome
    )


class TaskResponse(BaseModel):
    id: str
    status: str
    live_url: str


class TaskStatusResponse(BaseModel):
    status: str
    result: Optional[str] = None
    error: Optional[str] = None


# 从请求头获取用户ID的依赖项
async def get_user_id(x_user_id: Optional[str] = Header(None)) -> str:
    """从请求头提取用户ID或使用默认值"""
    return x_user_id or DEFAULT_USER_ID


# 实用函数
def get_llm(ai_provider: str):
    """根据提供商获取LLM"""
    logger.info(f"为提供商创建LLM: {ai_provider}")

    if ai_provider == "anthropic":
        return ChatAnthropic(
            model=os.environ.get("ANTHROPIC_MODEL_ID", "claude-3-5-sonnet-20240620")
        )
    elif ai_provider == "mistral":
        # MistralAI 通过 OpenAI 兼容 API 支持
        base_url = os.environ.get("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
        return ChatOpenAI(
            model=os.environ.get("MISTRAL_MODEL_ID", "mistral-large-latest"),
            base_url=base_url,
            api_key=os.environ.get("MISTRAL_API_KEY")
        )
    elif ai_provider == "google":
        model_id = os.environ.get("GOOGLE_MODEL_ID", "gemini-2.0-flash-exp")
        logger.info(f"使用Google AI模型: {model_id}")
        return ChatGoogle(model=model_id)
    elif ai_provider == "ollama":
        return ChatOllama(
            model=os.environ.get("OLLAMA_MODEL_ID", "llama3.1:8b")
        )
    elif ai_provider == "azure":
        return ChatAzureOpenAI(
            model=os.environ.get("AZURE_DEPLOYMENT_NAME", "gpt-4"),
        )
    elif ai_provider == "bedrock":
        return ChatAWSBedrock(
            model=os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0"),
            aws_region=os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        )
    elif ai_provider == "groq":
        return ChatGroq(
            model=os.environ.get("GROQ_MODEL_ID", "llama-3.3-70b-versatile")
        )
    else:  # 默认使用OpenAI
        logger.warning(f"使用OpenAI作为提供商的备选方案: {ai_provider}")
        base_url = os.environ.get("OPENAI_BASE_URL")
        kwargs = {"model": os.environ.get("OPENAI_MODEL_ID", "gpt-4o")}
        if base_url:
            kwargs["base_url"] = base_url
        logger.info(f"使用OpenAI配置: {kwargs}")
        return ChatOpenAI(**kwargs)


async def execute_task(task_id: str, instruction: str, ai_provider: str, user_id: str = DEFAULT_USER_ID):
    """在后台执行浏览器任务

    Chrome路径（CHROME_PATH和CHROME_USER_DATA）出于安全原因只能从
    环境变量中获取。
    """
    # 在try块外初始化浏览器变量
    browser = None
    try:
        # 更新任务状态
        task_storage.update_task_status(task_id, TaskStatus.RUNNING, user_id)

        # 获取LLM
        logger.info(f"任务 {task_id}: 使用提供商创建LLM: {ai_provider}")
        llm = get_llm(ai_provider)
        logger.info(f"任务 {task_id}: LLM创建成功: {type(llm).__name__}")

        # 获取任务特定的浏览器配置（如果可用）
        task = task_storage.get_task(task_id, user_id)
        task_browser_config = task.get("browser_config", {}) if task else {}

        # 提前创建任务媒体目录
        task_media_dir = MEDIA_DIR / task_id
        task_media_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"为任务 {task_id} 创建媒体目录: {task_media_dir}")

        # 配置浏览器无头/有头模式（任务设置覆盖环境变量）
        task_headful = task_browser_config.get("headful")
        if task_headful is not None:
            headful = task_headful
        else:
            headful = os.environ.get("BROWSER_USE_HEADFUL", "false").lower() == "true"

        # 获取Chrome路径和用户数据目录（任务设置覆盖环境变量）
        use_custom_chrome = task_browser_config.get("use_custom_chrome")

        if use_custom_chrome is False:
            # 为此任务明确禁用自定义Chrome
            chrome_path = None
            chrome_user_data = None
        else:
            # 仅使用环境变量中的Chrome路径
            chrome_path = os.environ.get("CHROME_PATH")
            chrome_user_data = os.environ.get("CHROME_USER_DATA")

        sensitive_data = {}
        for key, value in os.environ.items():
            if key.startswith("X_") and value:
                sensitive_data[key] = value

        # 配置代理选项 - 从基本配置开始
        agent_kwargs = {
            "task": instruction,
            "llm": llm,
            "sensitive_data": sensitive_data,
        }

        # 仅在需要自定义浏览器设置时配置和包含浏览器
        if not headful or chrome_path:
            extra_chromium_args = []
            # 配置浏览器
            browser_config_args = {
                "headless": not headful,
            }
            # 对于较旧的Chrome版本
            extra_chromium_args += ["--headless=new"]
            logger.info(
                f"任务 {task_id}: 浏览器配置参数: {browser_config_args.get('headless')}"
            )
            # 如果提供了Chrome可执行文件路径，则添加
            if chrome_path:
                browser_config_args["chrome_instance_path"] = chrome_path
                logger.info(
                    f"任务 {task_id}: 使用自定义Chrome可执行文件: {chrome_path}"
                )

            # 如果提供了Chrome用户数据目录，则添加
            if chrome_user_data:
                extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]
                logger.info(
                    f"任务 {task_id}: 使用Chrome用户数据目录: {chrome_user_data}"
                )

            browser_config = BrowserConfig(**browser_config_args)
            browser = Browser(config=browser_config)

            # 将浏览器添加到代理kwargs
            agent_kwargs["browser"] = browser

        logger.info(f"代理参数: {agent_kwargs}")
        # 将浏览器传递给代理
        agent = Agent(**agent_kwargs)

        # 在任务中存储代理
        task_storage.set_task_agent(task_id, agent, user_id)

        # 运行代理
        result = await agent.run(
            on_step_end=lambda agent_instance: automated_screenshot(agent_instance, task_id, user_id)
        )

        # 更新完成时间戳和任务状态
        task_storage.mark_task_finished(task_id, user_id, TaskStatus.FINISHED)

        # 提取结果
        if isinstance(result, AgentHistoryList):
            final_result = result.final_result()
            task_storage.set_task_output(task_id, final_result, user_id)
        else:
            task_storage.set_task_output(task_id, str(result), user_id)

        # 如果请求则收集浏览器数据
        task = task_storage.get_task(task_id, user_id)
        if task and task.get("save_browser_data") and hasattr(agent, "browser"):
            try:
                # 尝试多种方法收集浏览器数据
                if hasattr(agent.browser, "get_cookies"):
                    # 如果可用则使用直接方法
                    cookies = await agent.browser.get_cookies()
                    task_storage.update_task(task_id, {"browser_data": {"cookies": cookies}}, user_id)
                elif hasattr(agent.browser, "page") and hasattr(
                    agent.browser.page, "cookies"
                ):
                    # 尝试Playwright的page.cookies()方法
                    cookies = await agent.browser.page.cookies()
                    task_storage.update_task(task_id, {"browser_data": {"cookies": cookies}}, user_id)
                elif hasattr(agent.browser, "context") and hasattr(
                    agent.browser.context, "cookies"
                ):
                    # 尝试Playwright的context.cookies()方法
                    cookies = await agent.browser.context.cookies()
                    task_storage.update_task(task_id, {"browser_data": {"cookies": cookies}}, user_id)
                else:
                    logger.warning(
                        f"没有已知的方法为任务 {task_id} 收集cookies"
                    )
                    task_storage.update_task(
                        task_id,
                        {"browser_data": {
                            "cookies": [],
                            "error": "没有可用的方法收集cookies"
                        }},
                        user_id
                    )
            except Exception as e:
                logger.error(f"收集浏览器数据失败: {str(e)}")
                task_storage.update_task(
                    task_id,
                    {"browser_data": {"cookies": [], "error": str(e)}},
                    user_id
                )

    except Exception as e:
        logger.exception(f"执行任务 {task_id} 时出错")
        task_storage.update_task_status(task_id, TaskStatus.FAILED, user_id)
        task_storage.set_task_error(task_id, str(e), user_id)
        task_storage.mark_task_finished(task_id, user_id, TaskStatus.FAILED)
    finally:
        # 无论成功或失败都关闭浏览器
        if browser is not None:
            logger.info(f"关闭任务 {task_id} 的浏览器")
            try:
                logger.info(
                    f"任务 {task_id} 完成后截取最终截图"
                )

                # 获取代理进行截图
                agent = task_storage.get_task_agent(task_id, user_id)
                if agent and hasattr(agent, "browser_context"):
                    # 使用代理的浏览器上下文截图
                    screenshot_b64 = None
                    try:
                        pages = agent.browser_context.pages
                        if pages:
                            current_page = pages[-1]
                            # 使用页面的 screenshot 方法
                            screenshot_data = await current_page.screenshot(full_page=True)
                            # 转换为 base64
                            screenshot_b64 = base64.b64encode(screenshot_data).decode('utf-8')
                        else:
                            logger.warning("没有可用于截图的页面")
                    except Exception as e:
                        logger.warning(f"截图时出错: {str(e)}")
                        screenshot_b64 = None

                    if screenshot_b64:
                        # 如果不存在则创建任务媒体目录
                        task_media_dir = MEDIA_DIR / task_id
                        task_media_dir.mkdir(exist_ok=True, parents=True)

                        # 保存截图
                        screenshot_filename = "final_result.png"
                        screenshot_path = task_media_dir / screenshot_filename

                        # 解码base64并保存为文件
                        try:
                            image_data = base64.b64decode(screenshot_b64)
                            with open(screenshot_path, "wb") as f:
                                f.write(image_data)

                            # 验证文件已创建且有内容
                            if (
                                screenshot_path.exists()
                                and screenshot_path.stat().st_size > 0
                            ):
                                logger.info(
                                    f"最终截图已保存: {screenshot_path} ({screenshot_path.stat().st_size} 字节)"
                                )

                                # 添加到媒体列表中并指定类型
                                screenshot_url = f"/media/{task_id}/{screenshot_filename}"
                                media_entry = {
                                    "url": screenshot_url,
                                    "type": "screenshot",
                                    "filename": screenshot_filename,
                                    "created_at": datetime.now(UTC).isoformat() + "Z",
                                }
                                task_storage.add_task_media(task_id, media_entry, user_id)
                        except Exception as e:
                            logger.error(f"保存最终截图时出错: {str(e)}")
            except Exception as e:
                logger.error(f"截取最终截图时出错: {str(e)}")
                await browser.close()
                logger.info(f"任务 {task_id} 的浏览器关闭成功")
            except Exception as e:
                logger.error(f"关闭任务 {task_id} 的浏览器时出错: {str(e)}")


# API路由
@app.post("/api/v1/run-task", response_model=TaskResponse)
async def run_task(request: TaskRequest, user_id: str = Depends(get_user_id)):
    """启动浏览器自动化任务"""
    task_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat() + "Z"

    # 如果未提供则设置默认AI提供商
    ai_provider = request.ai_provider or os.environ.get("DEFAULT_AI_PROVIDER", "openai")
    logger.info(f"任务 {task_id}: 使用AI提供商: {ai_provider} (请求: {request.ai_provider}, 环境默认: {os.environ.get('DEFAULT_AI_PROVIDER', 'NOT_SET')})")

    # 生成实时URL
    live_url = f"/live/{task_id}"

    # 初始化任务记录
    task_data = {
        "id": task_id,
        "task": request.task,
        "ai_provider": ai_provider,
        "status": TaskStatus.CREATED,
        "created_at": now,
        "finished_at": None,
        "output": None,  # 最终结果
        "error": None,
        "steps": [],  # 将存储步骤信息
        "agent": None,
        "save_browser_data": request.save_browser_data,
        "browser_data": None,  # 如果请求将存储浏览器cookies
        # 存储浏览器配置选项
        "browser_config": {
            "headful": request.headful,
            "use_custom_chrome": request.use_custom_chrome,
        },
        "live_url": live_url,
    }

    # 将任务存储在存储中
    task_storage.create_task(task_id, task_data, user_id)

    # 在后台启动任务
    asyncio.create_task(execute_task(task_id, request.task, ai_provider, user_id))

    return TaskResponse(id=task_id, status=TaskStatus.CREATED, live_url=live_url)


async def automated_screenshot(agent, task_id, user_id=DEFAULT_USER_ID):
    try:
        # 修复：使用新的API获取当前页面
        pages = agent.browser_context.pages
        current_page = pages[-1] if pages else None

        if current_page:
            visit_log = agent.state.history.urls()
            current_url = current_page.url
            previous_url = visit_log[-2] if len(visit_log) >= 2 else None
            logger.info(f"代理上次在URL: {previous_url}，现在在 {current_url}")
        else:
            logger.warning("无法获取当前页面")

        await capture_screenshot(agent, task_id, user_id)
    except Exception as e:
        logger.error(f"自动截图错误: {str(e)}")
        # 仍然尝试截图，即使无法获取页面信息
        await capture_screenshot(agent, task_id, user_id)


@app.get("/api/v1/task/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: str, user_id: str = Depends(get_user_id)):
    """获取任务状态"""
    task = task_storage.get_task(task_id, user_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # 仅对运行中的任务增加步骤
    if task["status"] == TaskStatus.RUNNING:
        # 如果不存在则初始化步骤数组
        current_step = len(task.get("steps", [])) + 1

        # 添加步骤信息
        step_info = {
            "step": current_step,
            "timestamp": datetime.now(UTC).isoformat() + "Z",
            "next_goal": f"进度检查 {current_step}",
            "evaluation_previous_goal": "进行中",
        }

        task_storage.add_task_step(task_id, step_info, user_id)
        logger.info(f"为任务 {task_id} 添加步骤 {current_step}")

    await capture_screenshot(task_storage.get_task_agent(task_id, user_id), task_id, user_id)

    return TaskStatusResponse(
        status=task["status"],
        result=task.get("output"),
        error=task.get("error"),
    )


async def capture_screenshot(agent, task_id, user_id=DEFAULT_USER_ID):
    """从代理的浏览器上下文捕获截图"""
    logger.info(f"为任务捕获截图: {task_id}")

    # 检查代理是否存在
    if agent is None:
        logger.warning(f"任务 {task_id} 没有可用的代理")
        return

    # 记录代理类型以便调试
    logger.info(f"代理类型: {type(agent).__name__}")

    # 检查代理是否有browser_context
    if hasattr(agent, "browser_context") and agent.browser_context:
        try:
            logger.info(f"为任务截图: {task_id}")

            # 获取当前URL用于记录
            current_url = "unknown"
            try:
                pages = agent.browser_context.pages
                page = pages[-1] if pages else None
                current_url = page.url if page else "unknown"
            except Exception as e:
                logger.warning(f"无法获取当前页面: {str(e)}")
                page = None

            logger.info(f"截图前代理的当前URL: {current_url}")

            # 跳过捕获about:blank页面
            if current_url == "about:blank":
                logger.info("跳过截图 - 检测到空白页面")
                return

            # 使用代理的浏览器上下文截图
            screenshot_b64 = None
            try:
                pages = agent.browser_context.pages
                if pages:
                    current_page = pages[-1]
                    # 使用页面的screenshot方法
                    screenshot_data = await current_page.screenshot(full_page=True)
                    # 转换为base64
                    screenshot_b64 = base64.b64encode(screenshot_data).decode('utf-8')
                else:
                    logger.warning("没有可用于截图的页面")
            except Exception as e:
                logger.warning(f"截图时出错: {str(e)}")
                screenshot_b64 = None

            if screenshot_b64:
                # 使用适当的文件处理保存截图
                task_media_dir = MEDIA_DIR / task_id
                task_media_dir.mkdir(exist_ok=True, parents=True)

                # 生成基于时间戳的唯一名称
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

                # 更可靠地获取当前步骤号
                task = task_storage.get_task(task_id, user_id)
                if task and "steps" in task and task["steps"]:
                    # 使用最新的步骤号
                    current_step = (
                        task["steps"][-1]["step"] - 1
                    )  # 第一页始终为空白
                else:
                    current_step = "initial"

                # 根据截图时间生成文件名
                if task and task["status"] == TaskStatus.FINISHED:
                    screenshot_filename = f"final-{timestamp}.png"
                elif task and task["status"] == TaskStatus.RUNNING:
                    screenshot_filename = f"status-step-{current_step}-{timestamp}.png"
                else:
                    task_status = task["status"] if task else "unknown"
                    screenshot_filename = f"status-{task_status}-{timestamp}.png"

                screenshot_path = task_media_dir / screenshot_filename

                try:
                    image_data = base64.b64decode(screenshot_b64)
                    with open(screenshot_path, "wb") as f:
                        f.write(image_data)

                    if screenshot_path.exists() and screenshot_path.stat().st_size > 0:
                        logger.info(
                            f"截图已保存: {screenshot_path} ({screenshot_path.stat().st_size} 字节)"
                        )

                        screenshot_url = f"/media/{task_id}/{screenshot_filename}"
                        media_entry = {
                            "url": screenshot_url,
                            "type": "screenshot",
                            "filename": screenshot_filename,
                            "created_at": datetime.now(UTC).isoformat() + "Z",
                        }
                        task_storage.add_task_media(task_id, media_entry, user_id)
                    else:
                        logger.error(
                            f"截图文件未创建或为空: {screenshot_path}"
                        )
                except Exception as e:
                    logger.error(f"保存截图时出错: {str(e)}")
            else:
                logger.warning(f"截图不可用")
        except Exception as e:
            logger.error(f"截图时出错: {str(e)}")
    else:
        logger.warning(f"任务 {task_id} 的代理没有browser_context")


@app.get("/api/v1/task/{task_id}", response_model=dict)
async def get_task(task_id: str, user_id: str = Depends(get_user_id)):
    """获取完整任务详情"""
    task = task_storage.get_task(task_id, user_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return task


@app.put("/api/v1/stop-task/{task_id}")
async def stop_task(task_id: str, user_id: str = Depends(get_user_id)):
    """停止运行中的任务"""
    task = task_storage.get_task(task_id, user_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task["status"] in [
        TaskStatus.FINISHED,
        TaskStatus.FAILED,
        TaskStatus.STOPPED,
    ]:
        return {
            "message": f"任务已处于终端状态: {task['status']}"
        }

    # 获取代理
    agent = task_storage.get_task_agent(task_id, user_id)
    if agent:
        # 调用代理的停止方法
        agent.stop()
        task_storage.update_task_status(task_id, TaskStatus.STOPPING, user_id)
        return {"message": "任务正在停止"}
    else:
        task_storage.update_task_status(task_id, TaskStatus.STOPPED, user_id)
        task_storage.mark_task_finished(task_id, user_id, TaskStatus.STOPPED)
        return {"message": "任务已停止（未找到代理）"}


@app.put("/api/v1/pause-task/{task_id}")
async def pause_task(task_id: str, user_id: str = Depends(get_user_id)):
    """暂停运行中的任务"""
    task = task_storage.get_task(task_id, user_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task["status"] != TaskStatus.RUNNING:
        return {"message": f"任务未运行: {task['status']}"}

    # 获取代理
    agent = task_storage.get_task_agent(task_id, user_id)
    if agent:
        # 调用代理的暂停方法
        agent.pause()
        task_storage.update_task_status(task_id, TaskStatus.PAUSED, user_id)
        return {"message": "任务已暂停"}
    else:
        return {"message": "任务无法暂停（未找到代理）"}


@app.put("/api/v1/resume-task/{task_id}")
async def resume_task(task_id: str, user_id: str = Depends(get_user_id)):
    """恢复暂停的任务"""
    task = task_storage.get_task(task_id, user_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task["status"] != TaskStatus.PAUSED:
        return {"message": f"任务未暂停: {task['status']}"}

    # 获取代理
    agent = task_storage.get_task_agent(task_id, user_id)
    if agent:
        # 调用代理的恢复方法
        agent.resume()
        task_storage.update_task_status(task_id, TaskStatus.RUNNING, user_id)
        return {"message": "任务已恢复"}
    else:
        return {"message": "任务无法恢复（未找到代理）"}


@app.get("/api/v1/list-tasks")
async def list_tasks(
    user_id: str = Depends(get_user_id),
    page: int = Query(1, ge=1),
    per_page: int = Query(100, ge=1, le=1000)
):
    """列出所有任务"""
    return task_storage.list_tasks(user_id, page, per_page)


@app.get("/live/{task_id}", response_class=HTMLResponse)
async def live_view(task_id: str, user_id: str = Depends(get_user_id)):
    """获取可以嵌入iframe的任务实时视图"""
    task = task_storage.get_task(task_id, user_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Browser Use Task {task_id}</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .status {{ padding: 10px; border-radius: 4px; margin-bottom: 20px; }}
            .{TaskStatus.RUNNING} {{ background-color: #e3f2fd; }}
            .{TaskStatus.FINISHED} {{ background-color: #e8f5e9; }}
            .{TaskStatus.FAILED} {{ background-color: #ffebee; }}
            .{TaskStatus.PAUSED} {{ background-color: #fff8e1; }}
            .{TaskStatus.STOPPED} {{ background-color: #eeeeee; }}
            .{TaskStatus.CREATED} {{ background-color: #f3e5f5; }}
            .{TaskStatus.STOPPING} {{ background-color: #fce4ec; }}
            .controls {{ margin-bottom: 20px; }}
            button {{ padding: 8px 16px; margin-right: 10px; cursor: pointer; }}
            pre {{ background-color: #f5f5f5; padding: 15px; border-radius: 4px; overflow: auto; }}
            .step {{ margin-bottom: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Browser Use Task</h1>
            <div id="status" class="status">Loading...</div>

            <div class="controls">
                <button id="pauseBtn">Pause</button>
                <button id="resumeBtn">Resume</button>
                <button id="stopBtn">Stop</button>
            </div>

            <h2>Result</h2>
            <pre id="result">Loading...</pre>

            <h2>Steps</h2>
            <div id="steps">Loading...</div>

            <script>
                const taskId = '{task_id}';
                const FINISHED = '{TaskStatus.FINISHED}';
                const FAILED = '{TaskStatus.FAILED}';
                const STOPPED = '{TaskStatus.STOPPED}';
                const userId = '{user_id}';

                // Set user ID in request headers if available
                const headers = {{}};
                if (userId && userId !== 'default') {{
                    headers['X-User-ID'] = userId;
                }}

                // Update status function
                function updateStatus() {{
                    fetch(`/api/v1/task/${{taskId}}/status`, {{ headers }})
                        .then(response => response.json())
                        .then(data => {{
                            // Update status element
                            const statusEl = document.getElementById('status');
                            statusEl.textContent = `Status: ${{data.status}}`;
                            statusEl.className = `status ${{data.status}}`;

                            // Update result if available
                            if (data.result) {{
                                document.getElementById('result').textContent = data.result;
                            }} else if (data.error) {{
                                document.getElementById('result').textContent = `Error: ${{data.error}}`;
                            }}

                            // Continue polling if not in terminal state
                            if (![FINISHED, FAILED, STOPPED].includes(data.status)) {{
                                setTimeout(updateStatus, 2000);
                            }}
                        }})
                        .catch(error => {{
                            console.error('Error fetching status:', error);
                            setTimeout(updateStatus, 5000);
                        }});

                    // Also fetch full task to get steps
                    fetch(`/api/v1/task/${{taskId}}`, {{ headers }})
                        .then(response => response.json())
                        .then(data => {{
                            if (data.steps && data.steps.length > 0) {{
                                const stepsHtml = data.steps.map(step => `
                                    <div class="step">
                                        <strong>Step ${{step.step}}</strong>
                                        <p>Next Goal: ${{step.next_goal || 'N/A'}}</p>
                                        <p>Evaluation: ${{step.evaluation_previous_goal || 'N/A'}}</p>
                                    </div>
                                `).join('');
                                document.getElementById('steps').innerHTML = stepsHtml;
                            }} else {{
                                document.getElementById('steps').textContent = 'No steps recorded yet.';
                            }}
                        }})
                        .catch(error => {{
                            console.error('Error fetching task details:', error);
                        }});
                }}

                // Setup control buttons
                document.getElementById('pauseBtn').addEventListener('click', () => {{
                    fetch(`/api/v1/pause-task/${{taskId}}`, {{
                        method: 'PUT',
                        headers
                    }})
                        .then(response => response.json())
                        .then(data => alert(data.message))
                        .catch(error => console.error('Error pausing task:', error));
                }});

                document.getElementById('resumeBtn').addEventListener('click', () => {{
                    fetch(`/api/v1/resume-task/${{taskId}}`, {{
                        method: 'PUT',
                        headers
                    }})
                        .then(response => response.json())
                        .then(data => alert(data.message))
                        .catch(error => console.error('Error resuming task:', error));
                }});

                document.getElementById('stopBtn').addEventListener('click', () => {{
                    if (confirm('Are you sure you want to stop this task? This action cannot be undone.')) {{
                        fetch(`/api/v1/stop-task/${{taskId}}`, {{
                            method: 'PUT',
                            headers
                        }})
                            .then(response => response.json())
                            .then(data => alert(data.message))
                            .catch(error => console.error('Error stopping task:', error));
                    }}
                }});

                // Start status updates
                updateStatus();

                // Refresh every 5 seconds
                setInterval(updateStatus, 5000);
            </script>
        </div>
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)


@app.get("/api/v1/ping")
async def ping():
    """健康检查端点"""
    return {"status": "success", "message": "API运行正常"}


@app.get("/api/v1/debug/config")
async def debug_config():
    """调试端点用于检查配置"""
    return {
        "default_ai_provider": os.environ.get("DEFAULT_AI_PROVIDER", "NOT_SET"),
        "google_api_key_configured": bool(os.environ.get("GOOGLE_API_KEY")),
        "openai_api_key_configured": bool(os.environ.get("OPENAI_API_KEY")),
        "browser_headful": os.environ.get("BROWSER_USE_HEADFUL", "false"),
        "available_providers": ["openai", "google", "anthropic", "mistral", "ollama", "azure", "bedrock", "groq"]
    }


@app.get("/api/v1/browser-config")
async def browser_config():
    """获取当前浏览器配置

    注意：Chrome路径（CHROME_PATH和CHROME_USER_DATA）出于安全原因只能通过
    环境变量设置，不能在任务请求中覆盖。
    """
    headful = os.environ.get("BROWSER_USE_HEADFUL", "false").lower() == "true"
    chrome_path = os.environ.get("CHROME_PATH", None)
    chrome_user_data = os.environ.get("CHROME_USER_DATA", None)

    return {
        "headful": headful,
        "headless": not headful,
        "chrome_path": chrome_path,
        "chrome_user_data": chrome_user_data,
        "using_custom_chrome": chrome_path is not None,
        "using_user_data": chrome_user_data is not None,
    }


@app.get("/api/v1/task/{task_id}/media")
async def get_task_media(task_id: str, user_id: str = Depends(get_user_id), type: Optional[str] = None):
    """返回任务执行期间生成的任何录制或媒体的链接"""
    task = task_storage.get_task(task_id, user_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # 检查任务是否已完成
    if task["status"] not in [
        TaskStatus.FINISHED,
        TaskStatus.FAILED,
        TaskStatus.STOPPED,
    ]:
        raise HTTPException(
            status_code=400, detail="Media only available for completed tasks"
        )

    # 检查媒体目录是否存在并包含文件
    task_media_dir = MEDIA_DIR / task_id
    media_files = []

    if task_media_dir.exists():
        media_files = list(task_media_dir.glob("*"))
        logger.info(
            f"任务 {task_id} 的媒体目录包含 {len(media_files)} 个文件: {[f.name for f in media_files]}"
        )
    else:
        logger.warning(f"任务 {task_id} 的媒体目录不存在")

    # 如果有文件但没有媒体条目，现在创建它们
    if media_files and (
        not task.get("media") or len(task.get("media", [])) == 0
    ):
        for file_path in media_files:
            if file_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                file_url = f"/media/{task_id}/{file_path.name}"
                media_entry = {
                    "url": file_url,
                    "type": "screenshot",
                    "filename": file_path.name
                }
                task_storage.add_task_media(task_id, media_entry, user_id)

    # 获取带有媒体的更新任务
    task = task_storage.get_task(task_id, user_id)
    media_list = task.get("media", [])

    # 如果指定则按类型过滤
    if type and isinstance(media_list, list):
        if all(isinstance(item, dict) for item in media_list):
            # 包含类型信息的字典格式
            media_list = [item for item in media_list if item.get("type") == type]
            recordings = [item["url"] for item in media_list]
        else:
            # 只有URL没有类型信息
            recordings = []
            logger.warning(
                f"任务 {task_id} 的媒体列表不包含类型信息"
            )
    else:
        # 返回所有媒体
        if isinstance(media_list, list):
            if media_list and all(isinstance(item, dict) for item in media_list):
                recordings = [item["url"] for item in media_list]
            else:
                recordings = media_list
        else:
            recordings = []

    logger.info(f"为任务 {task_id} 返回 {len(recordings)} 个媒体项目")
    return {"recordings": recordings}


@app.get("/api/v1/task/{task_id}/media/list")
async def list_task_media(task_id: str, user_id: str = Depends(get_user_id), type: Optional[str] = None):
    """返回与任务相关的媒体文件的详细信息"""
    # 检查媒体目录是否存在
    task_media_dir = MEDIA_DIR / task_id

    if not task_storage.task_exists(task_id, user_id):
        raise HTTPException(status_code=404, detail="Task not found")

    if not task_media_dir.exists():
        return {
            "media": [],
            "count": 0,
            "message": f"未找到任务 {task_id} 的媒体",
        }

    media_info = []

    media_files = list(task_media_dir.glob("*"))
    logger.info(f"为任务 {task_id} 找到 {len(media_files)} 个媒体文件")

    for file_path in media_files:
        # 根据文件扩展名确定媒体类型
        file_type = "unknown"
        if file_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            file_type = "screenshot"
        elif file_path.suffix.lower() in [".mp4", ".webm"]:
            file_type = "recording"

        # 获取文件统计信息
        stats = file_path.stat()

        file_info = {
            "filename": file_path.name,
            "type": file_type,
            "size_bytes": stats.st_size,
            "created_at": datetime.fromtimestamp(stats.st_ctime).isoformat(),
            "url": f"/media/{task_id}/{file_path.name}",
        }
        media_info.append(file_info)

    # 如果指定则按类型过滤
    if type:
        media_info = [item for item in media_info if item["type"] == type]

    logger.info(f"为任务 {task_id} 返回 {len(media_info)} 个媒体项目")
    return {"media": media_info, "count": len(media_info)}


@app.get("/api/v1/media/{task_id}/{filename}")
async def get_media_file(
    task_id: str,
    filename: str,
    download: bool = Query(
        False, description="强制下载而不是在浏览器中查看"
    ),
):
    """提供媒体文件，支持查看或下载选项"""
    # 构造文件路径
    file_path = MEDIA_DIR / task_id / filename

    # 检查文件是否存在
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Media file not found")

    # 确定内容类型
    content_type, _ = mimetypes.guess_type(file_path)

    # 根据下载首选项设置标头
    headers = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    else:
        headers["Content-Disposition"] = f'inline; filename="{filename}"'

    # 返回带有适当标头的文件
    return FileResponse(
        path=file_path, media_type=content_type, headers=headers, filename=filename
    )


@app.get("/api/v1/test-screenshot")
async def test_screenshot(ai_provider: str = "google"):
    """测试端点，使用BrowserContext验证截图功能"""
    logger.info(f"使用提供商测试截图功能: {ai_provider}")

    browser_service = None
    browser_context = None

    try:
        # 配置浏览器
        headful = os.environ.get("BROWSER_USE_HEADFUL", "false").lower() == "true"
        browser_config_args = {"headless": not headful}

        # 如果提供了Chrome可执行文件路径，则添加
        chrome_path = os.environ.get("CHROME_PATH")
        if chrome_path:
            browser_config_args["chrome_instance_path"] = chrome_path

        logger.info(f"使用配置创建浏览器: {browser_config_args}")
        browser_config = BrowserConfig(**browser_config_args)
        browser_service = Browser(config=browser_config)

        # 创建具有take_screenshot方法的BrowserContext实例
        browser_context = BrowserContext(browser=browser_service)

        # 启动上下文并导航到example.com
        async with browser_context:
            logger.info("BrowserContext已创建，导航到example.com")
            await browser_context.navigate_to("https://example.com")

            # 现在对上下文调用take_screenshot
            logger.info("使用browser_context.take_screenshot进行截图")
            screenshot_b64 = await browser_context.take_screenshot(full_page=True)

            if not screenshot_b64:
                return {"error": "截图返回None或空字符串"}

            logger.info(f"截图已捕获: {len(screenshot_b64)} 字节")

            # 创建测试目录并保存截图
            test_dir = MEDIA_DIR / "test"
            test_dir.mkdir(exist_ok=True, parents=True)
            screenshot_path = test_dir / "test_screenshot.png"

            try:
                # 解码并保存截图
                image_data = base64.b64decode(screenshot_b64)
                logger.info(f"解码的base64数据: {len(image_data)} 字节")

                with open(screenshot_path, "wb") as f:
                    f.write(image_data)

                if screenshot_path.exists():
                    file_size = screenshot_path.stat().st_size
                    logger.info(
                        f"截图已保存: {screenshot_path} ({file_size} 字节)"
                    )

                    return {
                        "success": True,
                        "message": "截图已成功捕获并保存",
                        "file_size": file_size,
                        "file_path": str(screenshot_path),
                        "url": f"/media/test/test_screenshot.png",
                        "working_method": "browser_context.take_screenshot",
                    }
                else:
                    return {"error": "文件未创建"}
            except Exception as e:
                logger.exception("保存截图时出错")
                return {"error": f"保存截图时出错: {str(e)}"}
    except Exception as e:
        logger.exception("截图测试中出错")
        return {"error": f"测试失败: {str(e)}"}
    finally:
        # 清理资源
        if browser_context:
            try:
                await browser_context.close()
            except Exception as e:
                logger.warning(f"关闭浏览器上下文时出错: {str(e)}")

        if browser_service:
            try:
                await browser_service.close()
            except Exception as e:
                logger.warning(f"关闭浏览器时出错: {str(e)}")


# 如果直接执行则运行服务器
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
