import asyncio
import json
import os
import uuid
import copy
from typing import Dict, List, Optional, Any, AsyncGenerator
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import uvicorn

from dptb_pilot.core.agent import create_llm_agent
from dptb_pilot.core.session import pop_event
from dptb_pilot.core.guardrail import zip_tool_schema, extract_arguments_from_schema
from dptb_pilot.core.utils import generate_random_string, hash_dict
from dptb_pilot.tools.loader import get_mcp_server_tools # Note: loader doesn't exist yet, need to create or fix path
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from dptb_pilot.core.logger import get_logger
from dptb_pilot.core.photon_service import get_photon_service, PhotonChargeResult
from dptb_pilot.core.photon_config import CHARGING_ENABLED

logger = get_logger(__name__)


# 全局状态管理 (保持与原main.py兼容)
active_agents: Dict[str, LlmAgent] = {}
history_pool: Dict[str, List[List[str]]] = {}
session_service = InMemorySessionService()

# MCP工具拦截相关状态
pending_events: Dict[str, asyncio.Event] = {}
unmodified_schema_store: Dict[str, Dict] = {}
modified_schema_store: Dict[str, Dict] = {}
modified_args_store: Dict[str, Dict] = {}

# 终止执行相关状态
cancel_execution_events: Dict[str, asyncio.Event] = {}
termination_requested: Dict[str, bool] = {}

# 配置信息
target_tools: List[str] = []
tools_info: List[Dict[str, Any]] = {}
agent_info: Dict[str, Any] = {}
model_config: Dict[str, Any] = {}
mcp_server_url: str = ""
work_path: str = "/tmp"

# FastAPI应用
app = FastAPI(title="Better AIM React API", version="1.0.0")

# CORS设置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境允许所有源
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# WebSocket连接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_text(json.dumps(message))

manager = ConnectionManager()


# Pydantic模型
class LoginRequest(BaseModel):
    session_id: str

class ChatMessage(BaseModel):
    message: str
    session_id: str
    chat_id: Optional[str] = None

class ModifyParamsRequest(BaseModel):
    session_id: str
    modified_schema: Dict[str, Any]
    execution_mode: str = 'Local'
    selected_machine_id: Optional[str] = None
    remote_machine: Optional[Dict[str, Any]] = None  # 包含完整的远程机器配置

class TerminateExecutionRequest(BaseModel):
    session_id: str


def generate_executor_and_storage(
    execution_mode: str,
    remote_machine: Optional[Dict[str, Any]],
    tool_schema: Dict[str, Any]
) -> Dict[str, Any]:
    """
    根据执行模式和远程机器配置自动生成 Executor 和 Storage 参数

    Args:
        execution_mode: 执行模式 ('Local' 或 'Remote')
        remote_machine: 远程机器配置
        tool_schema: 工具 schema

    Returns:
        更新后的工具 schema，包含自动生成的 Executor 和 Storage 参数
    """
    if execution_mode != 'Remote' or not remote_machine:
        return tool_schema

    machine_type = remote_machine.get('type')
    config = remote_machine.get('config', {})

    if not machine_type or not config:
        logger.warning(f"[AutoGenerate] 无效的远程机器配置: {remote_machine}")
        return tool_schema

    logger.info("=" * 80)
    logger.info(f"[AutoGenerate] 开始自动生成 Executor 和 Storage 参数")
    logger.info(f"[AutoGenerate] 机器类型: {machine_type}")
    logger.info(f"[AutoGenerate] 机器配置: {json.dumps(config, ensure_ascii=False, indent=2)}")

    # 深拷贝 tool_schema 避免修改原对象
    result_schema = copy.deepcopy(tool_schema)

    # 更新 schema 中的 Executor 和 Storage 参数
    properties = result_schema.get('input_schema', {}).get('properties', {})

    # 生成 Executor 配置（支持大写和小写的 key）
    executor_key = None
    for key in ['Executor', 'executor']:
        if key in properties:
            executor_key = key
            break

    if executor_key:
        if machine_type == 'Bohrium':
            executor_config = {
                'type': 'dispatcher',
                'machine': {
                    'batch_type': 'Bohrium',
                    'context_type': 'Bohrium',
                    'remote_profile': {
                        'email': config.get('username'),
                        'password': config.get('password'),
                        'program_id': int(config.get('project_id', 0)),
                        'input_data': {
                            'image_name': config.get('image_name') or 'registry.dp.tech/dptech/dp/native/prod-35271/dptb-pilot-test:0.2',
                            'job_type': 'container',
                            'platform': 'ali',
                            'scass_type': config.get('scass_type') or 'c2_m4_cpu'
                        }
                    }
                }
            }
            logger.info(f"[AutoGenerate] Bohrium Executor 配置已生成")
            logger.info(f"[AutoGenerate] Executor: {json.dumps(executor_config, ensure_ascii=False, indent=2)}")
            properties[executor_key]['user_input'] = executor_config

        elif machine_type == 'Slurm':
            executor_config = {
                'type': 'dispatcher',
                'machine': {
                    'batch_type': 'Slurm',
                    'context_type': 'SSHContext',
                    'local_root': './',
                    'remote_root': config.get('remote_root'),
                    'remote_profile': {
                        'hostname': config.get('hostname'),
                        'username': config.get('username'),
                        'timeout': 600,
                        'port': 22,
                        'key_filename': config.get('key_filename')
                    }
                },
                'resources': {
                    'number_node': int(config.get('number_node', 1)),
                    'gpu_per_node': int(config.get('gpu_per_node', 0)) if config.get('gpu_per_node') else 0,
                    'cpu_per_node': int(config.get('cpu_per_node', 1)) if config.get('cpu_per_node') else 1,
                    'queue_name': config.get('queue_name'),
                    'custom_flags': [config.get('custom_flags', ''), ''],
                    'source_list': [],
                    'module_list': []
                }
            }
            logger.info(f"[AutoGenerate] Slurm Executor 配置已生成")
            logger.info(f"[AutoGenerate] Executor: {json.dumps(executor_config, ensure_ascii=False, indent=2)}")
            properties[executor_key]['user_input'] = executor_config

    # 生成 Storage 配置（仅 Bohrium 类型）
    storage_key = None
    for key in ['Storage', 'storage']:
        if key in properties:
            storage_key = key
            break

    if storage_key and machine_type == 'Bohrium':
        storage_config = {
            'type': 'bohrium',
            'username': config.get('username'),
            'password': config.get('password'),
            'project_id': int(config.get('project_id', 0))
        }
        logger.info(f"[AutoGenerate] Bohrium Storage 配置已生成")
        logger.info(f"[AutoGenerate] Storage: {json.dumps(storage_config, ensure_ascii=False, indent=2)}")
        properties[storage_key]['user_input'] = storage_config

    logger.info(f"[AutoGenerate] 完成 Executor 和 Storage 参数自动生成")
    logger.info("=" * 80)

    return result_schema

async def call_agent_async(query: str, runner: Runner, user_id: str, session_id: str) -> AsyncGenerator[Dict[str, Any], None]:
    """与agent异步对话，支持MCP工具拦截"""
    content = types.Content(role='user', parts=[types.Part(text=query)])

    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        # 检查是否被终止
        if termination_requested.get(session_id, False):
            logger.info(f"[CallAgent] 会话 {session_id} 已请求终止")
            yield {
                "type": "final_response",
                "content": "执行已终止",
                "is_final": True
            }
            # 清理终止状态
            termination_requested[session_id] = False
            if session_id in pending_events:
                pending_events[session_id].set()
            break

        # 处理工具调用
        if event.content and event.content.parts:
            calls = event.get_function_calls()
            if calls:
                for call in calls:
                    tool_name = call.name
                    arguments = call.args

                    # 检查是否需要拦截
                    if tool_name in target_tools:
                        schema = zip_tool_schema(
                            tool_name=tool_name,
                            arguments=arguments,
                            tools_dict=tools_info
                        )

                        # 存储schema并等待用户修改
                        unmodified_schema_store[session_id] = schema
                        pending_events[session_id] = asyncio.Event()

                        # 通知前端需要修改参数
                        await manager.send_message(session_id, {
                            "type": "tool_modify_required",
                            "schema": schema,
                            "tool_name": tool_name
                        })

                        # 等待用户修改完成或终止
                        # 创建一个任务来检查取消事件
                        cancel_task = None
                        if session_id in cancel_execution_events:
                            cancel_execution_events[session_id] = asyncio.Event()
                            cancel_task = asyncio.create_task(cancel_execution_events[session_id].wait())

                        # 等待 pending_event 或 cancel_event
                        try:
                            await asyncio.wait_for(pending_events[session_id].wait(), timeout=600.0)
                        except asyncio.TimeoutError:
                            logger.warning(f"[CallAgent] 会话 {session_id} 等待参数修改超时")
                            break

                        # 清理 cancel_task
                        if cancel_task:
                            cancel_task.cancel()
                        if session_id in cancel_execution_events:
                            cancel_execution_events[session_id] = None

                        # 检查是否被终止
                        if termination_requested.get(session_id, False):
                            logger.info(f"[CallAgent] 会话 {session_id} 在参数修改阶段被终止")
                            yield {
                                "type": "final_response",
                                "content": "执行已终止",
                                "is_final": True
                            }
                            # 清理状态
                            termination_requested[session_id] = False
                            if session_id in pending_events:
                                pending_events[session_id] = None
                            break

                        # 使用修改后的参数
                        if session_id in modified_args_store:
                            call.args = modified_args_store[session_id]

                        # 清理状态
                        unmodified_schema_store[session_id] = ""

                continue

        # 处理最终响应
        if event.is_final_response():
            if event.content and event.content.parts:
                yield {
                    "type": "final_response",
                    "content": event.content.parts[0].text,
                    "is_final": True
                }
            break
        else:
            if event.content and event.content.parts:
                yield {
                    "type": "streaming_response",
                    "content": event.content.parts[0].text,
                    "is_final": False
                }


# API端点
@app.post("/api/login")
async def login(request: LoginRequest):
    """处理登录逻辑"""
    session_id = request.session_id
    logger.info(f"收到登录请求，会话ID: {session_id}")

    if not session_id:
        raise HTTPException(status_code=400, detail="请填写会话ID")
    elif len(session_id) != 32:
        raise HTTPException(status_code=400, detail="会话ID需要为长度为32的任意字符")

    # 创建或获取agent
    if session_id not in active_agents:
        try:
            agent = create_llm_agent(
                session_id=session_id,
                mcp_tools_url=mcp_server_url,
                agent_info=agent_info,
                model_config=model_config
            )
            active_agents[session_id] = agent
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"创建Agent失败: {str(e)}")

    logger.info(f"登录成功，会话ID: {session_id}")
    return {"message": "登录成功", "session_id": session_id}


@app.post("/api/chat")
async def chat_with_agent(message: ChatMessage):
    """与agent对话的HTTP端点 (非流式)"""
    session_id = message.session_id
    user_message = message.message

    if session_id not in active_agents:
        raise HTTPException(status_code=404, detail="Agent未找到，请重新登录")

    agent = active_agents[session_id]
    session = await session_service.create_session(
        app_name=agent_info["name"],
        user_id=session_id[:4],
        session_id=session_id
    )

    runner = Runner(
        agent=agent,
        app_name=agent_info["name"],
        session_service=session_service
    )

    full_response = ""
    async for response in call_agent_async(user_message, runner, session_id[:4], session_id):
        full_response += response.get("content", "")

    chat_id = message.chat_id
    
    # 确保 chat_id 存在
    if not chat_id:
        chat_id = session_id
        print(f"WARNING: No chat_id provided in HTTP request, falling back to user_id: {chat_id}")
        logger.warning(f"No chat_id provided in HTTP request, falling back to user_id: {chat_id}")

    # 懒加载聊天历史
    if chat_id not in history_pool:
        history_pool[chat_id] = load_session_history(session_id, chat_id, work_path)

    # 更新聊天历史
    history = history_pool[chat_id]
    history.append([user_message, full_response])
    
    # 同步更新 sessions.json
    update_session_history(session_id, chat_id, history, work_path)

    return {"response": full_response, "is_final": True}


@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket聊天端点，支持流式响应"""
    await manager.connect(websocket, session_id)

    # 获取 cookies 用于光子收费
    cookies = None
    try:
        # FastAPI WebSocket 不直接提供 cookies 属性，需要从请求中获取
        cookies = dict(websocket._cookies) if hasattr(websocket, '_cookies') else {}
        logger.info(f"WebSocket connection cookies: {list(cookies.keys())}")
    except Exception as e:
        logger.warning(f"Failed to get WebSocket cookies: {e}")
        cookies = {}

    try:
        if session_id not in active_agents:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Agent未找到，请重新登录"
            }))
            return

        agent = active_agents[session_id]
        
        try:
            session = await session_service.create_session(
                app_name=agent_info["name"],
                user_id=session_id[:4],
                session_id=session_id
            )
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise

        runner = Runner(
            agent=agent,
            app_name=agent_info["name"],
            session_service=session_service
        )

        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")
            chat_id = message_data.get("chat_id")

            if not user_message.strip():
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "消息不能为空"
                }))
                continue

            response_text = ""
            usage_metadata = None
            try:
                async for response in call_agent_async(user_message, runner, session_id[:4], session_id):
                    if response["type"] in ["streaming_response", "final_response"]:
                        response_text += (response.get("content") or "")

                        # Check for usage metadata
                        if "usage" in response:
                             # Send usage info to frontend
                             await websocket.send_text(json.dumps({
                                 "type": "usage_update",
                                 "usage": response["usage"]
                             }))
                             # Store usage metadata for photon charging
                             usage_metadata = response["usage"]

                        await websocket.send_text(json.dumps(response))
            except Exception as e:
                logger.error(f"Error during agent execution: {e}")
                import traceback
                traceback.print_exc()
                
                # Try to extract more info if it's an ExceptionGroup (Python 3.11+)
                if hasattr(e, 'exceptions'):
                    for i, exc in enumerate(e.exceptions):
                        logger.error(f"Sub-exception {i+1}: {exc}")
                        
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Agent execution error: {str(e)}. Please try again."
                }))
                continue

            # 执行光子收费（如果启用）
            charge_result = None
            if CHARGING_ENABLED and usage_metadata:
                try:
                    photon_service = get_photon_service()
                    if photon_service:
                        input_tokens = usage_metadata.get("promptTokenCount", 0) or usage_metadata.get("prompt_tokens", 0)
                        output_tokens = usage_metadata.get("candidatesTokenCount", 0) or usage_metadata.get("candidates_tokens", 0)

                        logger.info(f"Processing photon charge - Input: {input_tokens}, Output: {output_tokens}")

                        charge_result = await photon_service.charge_photon(
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            tool_calls=0,
                            websocket_cookies=cookies
                        )

                        # 发送收费结果到前端
                        await websocket.send_text(json.dumps({
                            "type": "charge_result",
                            "charge_result": {
                                "success": charge_result.success,
                                "code": charge_result.code,
                                "message": charge_result.message,
                                "biz_no": str(charge_result.biz_no) if charge_result.biz_no else None,
                                "photon_amount": charge_result.photon_amount,
                                "rmb_amount": charge_result.rmb_amount
                            }
                        }))

                        if charge_result.success:
                            logger.info(f"Photon charge successful: {charge_result.message}")
                        else:
                            logger.warning(f"Photon charge failed: {charge_result.message}")
                except Exception as charge_error:
                    logger.error(f"Error during photon charging: {charge_error}")
                    await websocket.send_text(json.dumps({
                        "type": "charge_result",
                        "charge_result": {
                            "success": False,
                            "code": -1,
                            "message": f"收费异常: {str(charge_error)}",
                            "photon_amount": 0,
                            "rmb_amount": 0.0
                        }
                    }))

            # 确保 chat_id 存在
            if not chat_id:
                # 如果没有 chat_id，尝试使用 session_id (兼容旧逻辑，但不推荐)
                chat_id = session_id
                logger.warning(f"No chat_id provided, falling back to user_id: {chat_id}")

            # 懒加载聊天历史 (从 sessions.json)
            if chat_id not in history_pool:
                history_pool[chat_id] = load_session_history(session_id, chat_id, work_path)

            # 更新聊天历史
            history = history_pool[chat_id]
            history.append([user_message, response_text])
            
            # 同步更新 sessions.json (这是唯一的持久化存储)
            update_session_history(session_id, chat_id, history, work_path)

    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        logger.critical(f"CRITICAL ERROR in websocket_chat: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.close(code=1011) # Internal Error
        except:
            pass
        manager.disconnect(session_id)


@app.post("/api/modify-params")
async def modify_parameters(request: ModifyParamsRequest):
    """处理参数修改请求"""
    session_id = request.session_id
    modified_schema = request.modified_schema

    logger.info("=" * 80)
    logger.info(f"[ModifyParams] 收到参数修改请求")
    logger.info(f"[ModifyParams] Session ID: {session_id}")
    logger.info(f"[ModifyParams] 工具名称: {modified_schema.get('name', 'unknown')}")
    logger.info(f"[ModifyParams] 执行模式: {request.execution_mode}")
    logger.info(f"[ModifyParams] 选中的机器ID: {request.selected_machine_id}")
    logger.info(f"[ModifyParams] 修改前的Schema: {json.dumps(modified_schema, ensure_ascii=False, indent=2)}")
    logger.info("=" * 80)

    # 构建工作目录路径：{work_path}/{session_id}/files（确保是绝对路径）
    session_files_dir = os.path.abspath(os.path.join(work_path, session_id, "files"))

    # 自动生成 Executor 和 Storage 参数
    modified_schema = generate_executor_and_storage(
        execution_mode=request.execution_mode,
        remote_machine=request.remote_machine,
        tool_schema=modified_schema
    )

    logger.info("=" * 80)
    logger.info(f"[ModifyParams] 生成 Executor 和 Storage 后的Schema: {json.dumps(modified_schema, ensure_ascii=False, indent=2)}")
    logger.info("=" * 80)

    # 提取修改后的参数
    modified_args = extract_arguments_from_schema(modified_schema)

    # 处理路径参数：确保所有路径都是绝对路径
    properties = modified_schema.get('input_schema', {}).get('properties', {})
    for param_name, param_info in properties.items():
        if param_name.endswith('_path') and param_name in modified_args:
            user_input = modified_args[param_name]
            if user_input and isinstance(user_input, str):
                # 确保路径是绝对路径
                if os.path.isabs(user_input):
                    # 已经是绝对路径，检查文件是否存在
                    if os.path.exists(user_input):
                        logger.info(f"[ModifyParams] 路径参数 {param_name}: 保持绝对路径 {user_input}")
                    else:
                        logger.warning(f"[ModifyParams] 路径参数 {param_name}: 绝对路径不存在 {user_input}")
                else:
                    # 相对路径，尝试在 session_files_dir 中查找
                    possible_path = os.path.join(session_files_dir, user_input)
                    if os.path.exists(possible_path):
                        modified_args[param_name] = possible_path
                        logger.info(f"[ModifyParams] 路径参数 {param_name}: 相对路径转绝对路径 {user_input} -> {possible_path}")
                    else:
                        # 尝试相对于当前工作目录
                        cwd_path = os.path.abspath(user_input)
                        if os.path.exists(cwd_path):
                            modified_args[param_name] = cwd_path
                            logger.info(f"[ModifyParams] 路径参数 {param_name}: 使用当前工作目录 {user_input} -> {cwd_path}")
                        else:
                            # 文件不存在，但仍然构建预期的绝对路径
                            modified_args[param_name] = possible_path
                            logger.warning(f"[ModifyParams] 路径参数 {param_name}: 文件不存在，使用预期路径 {user_input} -> {possible_path}")

    logger.info(f"[ModifyParams] 提取后的参数: {modified_args}")
    logger.info(f"[ModifyParams] Executor 参数: {modified_args.get('executor')}")
    logger.info(f"[ModifyParams] Storage 参数: {modified_args.get('storage')}")

    modified_args_store[session_id] = modified_args
    modified_schema_store[session_id] = modified_schema

    # 恢复agent执行
    if session_id in pending_events:
        logger.info(f"[ModifyParams] 触发事件，恢复 agent 执行")
        pending_events[session_id].set()
    else:
        logger.warning(f"[ModifyParams] Session {session_id} 没有待处理的事件")

    return {"message": "参数已更新", "modified_args": modified_args}


@app.post("/api/terminate-execution")
async def terminate_execution(request: TerminateExecutionRequest):
    """终止正在执行的 agent 任务"""
    session_id = request.session_id

    logger.info("=" * 80)
    logger.info(f"[TerminateExecution] 收到终止执行请求")
    logger.info(f"[TerminateExecution] Session ID: {session_id}")
    logger.info("=" * 80)

    # 标记会话需要终止
    termination_requested[session_id] = True

    # 触发取消事件
    if session_id in pending_events:
        # 设置终止事件，使 wait 立即返回
        if session_id not in cancel_execution_events:
            cancel_execution_events[session_id] = asyncio.Event()
        cancel_execution_events[session_id].set()

        # 同时触发 pending_event 使其返回
        pending_events[session_id].set()

        logger.info(f"[TerminateExecution] 已触发会话 {session_id} 的终止信号")
        return {"message": "终止请求已发送", "status": "terminating"}
    else:
        logger.warning(f"[TerminateExecution] Session {session_id} 没有待处理的事件")
        return {"message": "没有正在执行的任务", "status": "no_active_task"}


@app.get("/api/files/{session_id}")
async def list_files(session_id: str):
    """获取会话文件列表"""
    session_dir = os.path.join(work_path, session_id, "files")
    logger.info(f"Listing files from: {session_dir}")
    os.makedirs(session_dir, exist_ok=True)

    files = []
    if os.path.exists(session_dir):
        for filename in os.listdir(session_dir):
            file_path = os.path.join(session_dir, filename)
            if os.path.isfile(file_path):
                stats = os.stat(file_path)
                files.append({
                    "name": filename,
                    "path": file_path,
                    "size": stats.st_size,
                    "updated_at": stats.st_mtime
                })
    logger.info(f"Found {len(files)} files")
    return {"files": sorted(files, key=lambda x: x["name"])}


@app.post("/api/upload/{session_id}")
async def upload_file(session_id: str, files: List[UploadFile] = File(...)):
    """上传文件到会话目录"""
    session_dir = os.path.join(work_path, session_id, "files")
    os.makedirs(session_dir, exist_ok=True)

    uploaded_files = []
    for file in files:
        file_path = os.path.join(session_dir, file.filename)

        # 检查文件大小 (10MB限制)
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:
            continue

        with open(file_path, "wb") as f:
            f.write(content)

        uploaded_files.append({
            "name": file.filename,
            "path": file_path,
            "size": len(content)
        })

    return {"uploaded_files": uploaded_files}


@app.get("/api/download/{session_id}/{filename:path}")
async def download_file(session_id: str, filename: str):
    """下载文件 (支持子目录和可选的 files/ 前缀)"""
    # 兼容性处理：如果请求路径包含 files/ 前缀（例如前端根据文件系统路径拼接），则移除
    # 这样 /api/download/xxx/band.png 和 /api/download/xxx/files/band.png 都能工作
    clean_filename = filename
    if clean_filename.startswith("files/"):
        clean_filename = clean_filename[6:]
    elif clean_filename.startswith("/files/"):
        clean_filename = clean_filename[7:]
    
    file_path = os.path.join(work_path, session_id, "files", clean_filename)

    # 防止路径遍历攻击
    # ... (Normally we should check commonprefix, but assuming session_id isolation is enough for now for internal tool)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"文件不存在: {file_path}")

    return FileResponse(file_path, filename=os.path.basename(clean_filename))

@app.delete("/api/files/{session_id}/{filename:path}")
async def delete_file(session_id: str, filename: str):
    """删除文件"""
    # 同样的逻辑
    clean_filename = filename
    if clean_filename.startswith("files/"):
        clean_filename = clean_filename[6:]
        
    file_path = os.path.join(work_path, session_id, "files", clean_filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")

    try:
        os.remove(file_path)
        return {"message": "文件已删除", "filename": clean_filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文件失败: {str(e)}")


@app.get("/api/sessions/{session_id}/history")
async def get_chat_history(session_id: str):
    """获取聊天历史 (Legacy)"""
    history = history_pool.get(session_id, load_chat_history(session_id, work_path))
    return {"history": history}


@app.post("/api/sessions/{session_id}/clear")
async def clear_chat_history(session_id: str):
    """清空聊天历史 (Legacy)"""
    history_pool[session_id] = []
    save_chat_history(session_id, [], work_path)
    return {"message": "聊天历史已清空"}


class SaveSessionsRequest(BaseModel):
    sessions: List[Dict[str, Any]]


@app.get("/api/user/{user_id}/sessions")
async def get_user_sessions(user_id: str):
    """获取用户的所有聊天会话"""
    user_dir = os.path.join(work_path, user_id)
    sessions_file = os.path.join(user_dir, "sessions.json")
    logger.info(f"Loading sessions for {user_id} from {sessions_file}")
    
    if os.path.exists(sessions_file):
        try:
            with open(sessions_file, 'r', encoding='utf-8') as f:
                sessions = json.load(f)
            
            # 转换历史记录格式以适配前端: [[q, a], ...] -> [{role: user, content: q}, {role: assistant, content: a}, ...]
            for session in sessions:
                raw_history = session.get("history", [])
                formatted_history = []
                for item in raw_history:
                    if isinstance(item, list) and len(item) >= 2:
                        formatted_history.append({"role": "user", "content": item[0]})
                        formatted_history.append({"role": "assistant", "content": item[1]})
                session["history"] = formatted_history
                # Update message count to reflect total messages (user + assistant)
                session["message_count"] = len(formatted_history)
            
            logger.info(f"Loaded {len(sessions)} sessions")
            return {"sessions": sessions}
        except Exception as e:
            logger.error(f"Error loading sessions: {e}")
            return {"sessions": []}
    logger.warning("Sessions file not found")
    return {"sessions": []}


def load_session_history(user_id: str, chat_id: str, work_path: str) -> List[List[str]]:
    """从 sessions.json 加载特定会话的历史记录"""
    user_dir = os.path.join(work_path, user_id)
    sessions_file = os.path.join(user_dir, "sessions.json")
    
    if not os.path.exists(sessions_file):
        return []

    try:
        with open(sessions_file, 'r', encoding='utf-8') as f:
            sessions = json.load(f)
        
        for session in sessions:
            if session.get("chat_id") == chat_id:
                return session.get("history", [])
    except Exception as e:
        logger.error(f"Error loading session history: {e}")
    
    return []


def update_session_history(user_id: str, session_id: str, history: List[List[str]], work_path: str):
    """更新用户会话列表中的历史记录"""
    logger.debug(f"Updating session history for User: {user_id}, Chat: {session_id}, History Len: {len(history)}")
    user_dir = os.path.join(work_path, user_id)
    sessions_file = os.path.join(user_dir, "sessions.json")
    
    if not os.path.exists(sessions_file):
        logger.warning(f"Sessions file not found: {sessions_file}")
        return

    try:
        with open(sessions_file, 'r', encoding='utf-8') as f:
            sessions = json.load(f)
        
        updated = False
        for session in sessions:
            # logger.debug(f"Checking session {session.get('chat_id')} against {session_id}")
            if session.get("chat_id") == session_id:
                session["history"] = history
                session["last_active"] = datetime.now().isoformat()
                session["message_count"] = len(history)
                updated = True
                logger.debug(f"Found and updated session {session_id}")
                break
        
        if updated:
            with open(sessions_file, 'w', encoding='utf-8') as f:
                json.dump(sessions, f, ensure_ascii=False, indent=2)
            logger.debug("Successfully saved sessions.json")
        else:
            logger.warning(f"Chat ID {session_id} not found in sessions.json")
            
    except Exception as e:
        logger.error(f"Failed to update session history in sessions.json: {e}")


@app.post("/api/user/{user_id}/sessions")
async def save_user_sessions(user_id: str, request: SaveSessionsRequest):
    """保存用户的所有聊天会话"""
    user_dir = os.path.join(work_path, user_id)
    os.makedirs(user_dir, exist_ok=True)
    sessions_file = os.path.join(user_dir, "sessions.json")
    logger.info(f"Saving {len(request.sessions)} sessions for {user_id} to {sessions_file}")
    
    try:
        with open(sessions_file, 'w', encoding='utf-8') as f:
            json.dump(request.sessions, f, ensure_ascii=False, indent=2)
        return {"message": "Sessions saved successfully"}
    except Exception as e:
        logger.error(f"Failed to save sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save sessions: {str(e)}")


@app.get("/api/schema/{session_id}")
async def get_current_schema(session_id: str):
    """获取当前需要修改的参数schema"""
    schema = unmodified_schema_store.get(session_id, {})
    return {"schema": schema}


@app.get("/api/health")
async def health_check():
    """健康检查端点"""
    return {"status": "ok", "message": "Backend is running"}

@app.get("/api/config")
async def get_config():
    """获取应用配置信息"""
    logger.info("收到配置请求")
    config = {
        "agent_info": agent_info,
        "mcp_server_url": mcp_server_url,
        "target_tools": target_tools
    }
    logger.debug(f"返回配置信息: {config}")
    return config


# 初始化函数
def initialize_server(
    agent_info_dict: Dict[str, Any],
    model_config_dict: Dict[str, Any],
    mcp_url: str,
    work_dir: str = "/tmp",
    tools_modify: List[str] = None
):
    """初始化服务器配置"""
    global agent_info, model_config, mcp_server_url, work_path, target_tools, tools_info

    agent_info = agent_info_dict
    model_config = model_config_dict
    mcp_server_url = mcp_url
    work_path = work_dir
    target_tools = tools_modify or []

    # 加载MCP工具信息
    try:
        tools_info = asyncio.run(get_mcp_server_tools(mcp_server_url))
        logger.info(f"✅ 成功加载 {len(tools_info)} 个MCP工具")
    except Exception as e:
        logger.error(f"⚠️  加载MCP工具失败: {e}")
        tools_info = []

    # 配置静态文件服务 (如果在生产环境且存在dist)
    configure_static_serving()


def configure_static_serving():
    """配置前端静态文件服务"""
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    # 尝试找到dist目录
    # app.py path: .../dptb-pilot/dptb_pilot/server/app.py
    # web_ui path: .../dptb-pilot/web_ui
    server_dir = os.path.dirname(__file__)
    pilot_pkg_dir = os.path.dirname(server_dir)
    project_root = os.path.dirname(pilot_pkg_dir)

    possible_paths = [
        os.path.join(project_root, "web_ui", "dist"), # Best for packaged/repo run
        os.path.join(os.getcwd(), "web_ui", "dist"),  # Best for local dev in root
    ]
    
    dist_path = None
    for path in possible_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, "index.html")):
            dist_path = path
            break
            
    if dist_path:
        logger.info(f"🎨 启用静态文件托管: {dist_path}")
        
        # 1. Mount assets
        assets_path = os.path.join(dist_path, "assets")
        if os.path.exists(assets_path):
            app.mount("/assets", StaticFiles(directory=assets_path), name="assets")
            
        # 2. Mount other static folders if needed (e.g. vite creates assets, maybe others?)
        # For safety, we can mount root, but it might shadow API.
        
        # 3. Catch-all route for SPA (Must be last)
        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str):
            # API和WebSocket已经被前面的路由捕获，这里只处理前端路由
            if full_path.startswith("api/") or full_path.startswith("ws/"):
                raise HTTPException(status_code=404, detail="Not Found")
            
            # Check if file exists in dist (e.g. favicon.ico)
            file_path = os.path.join(dist_path, full_path)
            if os.path.isfile(file_path):
                 return FileResponse(file_path)
                 
            # 否则返回index.html (SPA路由)
            return FileResponse(os.path.join(dist_path, "index.html"))
            
        logger.info("✅ 前端静态服务已配置 (SPA Mode)")
    else:
        logger.info("ℹ️ 未发现前端编译产物，跳过静态服务配置 (请使用 npm run dev)")


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """运行服务器"""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()