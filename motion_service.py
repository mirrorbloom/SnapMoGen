import warnings
warnings.filterwarnings("ignore")

import os
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, Callable, Awaitable, Deque, List
from os.path import join as pjoin
from collections import deque
import re
import tempfile
import shutil
import threading

# Async HTTP and file helpers used by submit/check/download utilities
import aiohttp
import aiofiles
from pathlib import Path

# Simple logging helper for the module (used as Log in helper functions)
import logging

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from minio import Minio
from minio.error import S3Error
import io
from contextlib import asynccontextmanager
from openai import OpenAI

# Module logger used by existing helper code; replace/configure as needed.
Log = logging.getLogger("motion_service")
if not Log.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    Log.addHandler(handler)
Log.setLevel(logging.INFO)

def load_env_file(env_path: str) -> None:
    """Load environment variables from a .env-style file without overriding existing values."""
    candidate = Path(env_path)
    if not candidate.is_file():
        candidate = Path(__file__).resolve().parent / env_path
    if not candidate.is_file():
        return

    try:
        with candidate.open("r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        Log.info(f"Loaded environment variables from {candidate}")
    except Exception as exc:
        Log.warning(f"Failed to load environment file {candidate}: {exc}")


load_env_file(os.getenv("ENV_FILE_PATH", ".env.dev"))

# Optional external service API key used in submit_task helper; default to empty string if not set.
BLENDER_SERVICE_API_KEY = os.getenv("BLENDER_SERVICE_API_KEY", "")

# External Blender/retarget service endpoints
BLENDER_BASE_URL = os.getenv("BLENDER_BASE_URL", "http://localhost:8001")
RETARGET_SERVICE_URL = os.getenv("RETARGET_SERVICE_URL", f"{BLENDER_BASE_URL.rstrip('/')}/retarget")
RETARGET_STATUS_URL = os.getenv("RETARGET_STATUS_URL", f"{BLENDER_BASE_URL.rstrip('/')}/retarget_status")
RENDER_SERVICE_URL = os.getenv("RENDER_SERVICE_URL", f"{BLENDER_BASE_URL.rstrip('/')}/render_animation")
RENDER_STATUS_URL = os.getenv("RENDER_STATUS_URL", f"{BLENDER_BASE_URL.rstrip('/')}/render_status")
RETARGET_POLL_INTERVAL = float(os.getenv("RETARGET_POLL_INTERVAL", "2.0"))
RETARGET_PRESET_NAME = os.getenv("RETARGET_PRESET_NAME", "")
RETARGET_TARGET_FILE = os.getenv("RETARGET_TARGET_FILE", "")
RETARGET_SOURCE_REST_FILE = os.getenv("RETARGET_SOURCE_REST_FILE", "")
RETARGET_TARGET_REST_FILE = os.getenv("RETARGET_TARGET_REST_FILE", "")

# Asset paths
ASSET_PATH = os.getenv("ASSET_PATH", "./assets")
MOMAX_BOT_PATH = os.getenv("MOMAX_BOT_PATH", f"{ASSET_PATH}/MomaxBot.fbx")

from model.vq.rvq_model import HRVQVAE
from model.transformer.transformer import MoMaskPlus
from model.cnn_networks import GlobalRegressor
from config.load_config import load_config
from utils.fixseeds import fixseed
from utils import bvh_io
from utils.motion_process_bvh import process_bvh_motion, recover_bvh_from_rot
from utils.paramUtil import kinematic_chain
from common.skeleton import Skeleton
from common.quaternion import qeuler_np
import collections
from common.animation import Animation
from einops import rearrange, repeat
from rest_pose_retarget import RestPoseRetargeter

# Global variables for models and configurations
models = {}
cfg = None
skeleton = None
mean = None
std = None
retargeter = None

# Task status tracking
task_status: Dict[str, Dict[str, Any]] = {}
task_lock = threading.Lock()

# Task queue infrastructure
task_queue_dict: Dict[str, asyncio.Queue] = {}
pending_queue_map: Dict[str, Deque[str]] = {}
queue_workers: Dict[str, List[asyncio.Task]] = {}

# Queue configuration
QUEUE_DEFAULT_WORKERS = int(os.getenv("MOTION_QUEUE_DEFAULT_WORKERS", "1"))
NUM_WORKERS = int(os.getenv("MOTION_NUM_WORKERS", "1"))
POLL_INTERVAL_SEC = float(os.getenv("MOTION_POLL_INTERVAL_SEC", "5"))
TASK_STALE_MIN = float(os.getenv("MOTION_TASK_STALE_MIN", "60"))

# Task cleanup configuration
MAX_TASK_RETENTION_HOURS = 24  # Keep completed tasks for 24 hours
MAX_TASK_COUNT = 1000  # Maximum number of tasks to keep in memory

# MinIO client
minio_client = None

# OpenAI client for prompt rewriting (will be initialized from environment variables)
openai_client = None

# System prompts for LLM rewriting
SYSTEM_PROMPT_WITH_DURATION = """Task Description: Your task is to rewrite text prompts of user inputs for a text-to-motion generation model
inference. This model generates 3D human motion data from text, you need to understand the intent of the
user input and describe how the human body should move in detail.

Instructions:
1. Make sure the rewritten prompts describe the human motion without major information loss.
2. Be related to human body movements—the tool is not able to generate anything else.
3. The rewritten prompt should be around 60 words, no more than 100.
4. Use a clear, descriptive, and precise tone.
5. Be creative and make the motion interesting and expressive.
6. Feel free to add physical movement details.
7. The final output only contains the rewritten prompt without anything else. Output should be in English.

Example1:
Input:
Shooting a basketball.
Output:
The person stands neutrally, then leans forward, spreading their legs wide. They simulate
basketball dribbling with hand gestures, moving their hips side to side. The left hand performs dribbling
actions. They pause, turn left, put the right leg forward, and squat slightly before simulating a basketball
shot with a small jump.

Example2:
Input:
Zombie walk.
Output:
The person shuffles forward with a stiff, dragging motion, one foot scraping the ground as it
moves. His arms hang loosely by its sides, occasionally jerking forward as it staggers with uneven steps."""

SYSTEM_PROMPT_WITHOUT_DURATION = """Task Description: Your task is to rewrite text prompts of user inputs for a text-to-motion generation model
inference. This model generates 3D human motion data from text, you need to understand the intent of the
user input and describe how the human body should move in detail, and give me the proper duration of the
motion clip, usually from 2 to 12 seconds.

Instructions:
1. Make sure the rewritten prompts describe the human motion without major information loss.
2. Be related to human body movements—the tool is not able to generate anything else.
3. The rewritten prompt should be around 60 words, no more than 100.
4. Use a clear, descriptive, and precise tone.
5. Be creative and make the motion interesting and expressive.
6. Feel free to add physical movement details.
7. The final output only contains the rewritten prompt and duration without anything else. Output should be in English.

Example1:
Input:
Shooting a basketball.
Output:
The person stands neutrally, then leans forward, spreading their legs wide. They simulate
basketball dribbling with hand gestures, moving their hips side to side. The left hand performs dribbling
actions. They pause, turn left, put the right leg forward, and squat slightly before simulating a basketball
shot with a small jump.
Duration: 8 seconds

Example2:
Input:
Zombie walk.
Output:
The person shuffles forward with a stiff, dragging motion, one foot scraping the ground as it
moves. His arms hang loosely by its sides, occasionally jerking forward as it staggers with uneven steps.
Duration: 6 seconds"""

class MotionRequest(BaseModel):
    text: str
    duration: float  # in seconds
    
class TaskStatusResponse(BaseModel):
    task_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: Optional[float] = None
    message: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
    download_url: Optional[str] = None
    result_urls: Optional[Dict[str, str]] = None
    error_message: Optional[str] = None
    # Queue metadata
    queue_position: Optional[int] = None
    queue_size: Optional[int] = None
    queue_name: Optional[str] = None
    queued_at: Optional[str] = None
    last_progress_at: Optional[str] = None

class UploadResponse(BaseModel):
    task_id: str
    status: str
    message: str

async def submit_task(url: str, **kwargs) -> Dict:
    try:
        data = aiohttp.FormData()
        params = {}
        for key, value in kwargs.items():
            if value is None:
                continue
            file_path = Path(value) if isinstance(value, (str, Path)) else None
            if file_path and file_path.is_file():
                Log.info(f"上传文件 {key}: {file_path}")
                async with aiofiles.open(file_path, "rb") as f:
                    file_bytes = await f.read()
                data.add_field(
                    name=key,
                    value=file_bytes,
                    filename=file_path.name,
                    content_type="application/octet-stream",
                )
            else:
                if isinstance(value, bool):
                    payload = "true" if value else "false"
                else:
                    payload = str(value)
                Log.info(f"添加表单字段 {key}: {payload}")
                params[key] = payload
        url += "?" + "&".join([f"{k}={v}" for k, v in params.items()])
        headers = {}
        if BLENDER_SERVICE_API_KEY:
            headers["Authorization"] = f"Bearer {BLENDER_SERVICE_API_KEY}"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                # 尝试读取错误详情
                err_text = await response.text()
                Log.error(f"提交任务失败: {response.status}, body={err_text}")
                return {"error": str(response.status), "body": err_text}
    except Exception as e:
        Log.error(f"调用提交任务失败: {e}")
        return {"error": str(e)}

async def check_task_status(url: str, task_id: str) -> Dict:
    try:
        async with aiohttp.ClientSession() as session:
            # # 优先使用查询参数风格：/retarget_status?task_id=...
            # async with session.get(url, params={"task_id": task_id}) as response:
            #     if response.status == 200:
            #         return await response.json()
            #     fallback_text = await response.text()
            #     Log.warning(f"状态查询(参数)失败: {response.status}, body={fallback_text}, 尝试路径风格")
            # 优先路径参数风格：/retarget_status/<task_id>
            path_url = f"{url.rstrip('/')}/{task_id}"
            async with session.get(path_url) as response:
                if response.status == 200:
                    return await response.json()
                Log.error(f"检查任务状态失败: {response.status}")
                return {"error": str(response.status), "body": await response.text()}

    except Exception as e:
        Log.error(f"调用检查任务状态失败: {e}")
        return {"error": str(e)}

async def download_file(url: str, output_path: str) -> bool:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    with open(output_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                    return True
                else:
                    Log.error(f"下载文件失败: {response.status}")
                    return False
    except Exception as e:
        Log.error(f"调用下载文件失败: {e}")
        return False

async def retarget_to_smplx_async(source_bvh_url: str, output_fbx_path: str, preset_name: Optional[str] = None) -> bool:
    """Send BVH to external retarget service and save FBX output locally."""
    preset = preset_name or RETARGET_PRESET_NAME or None
    temp_dir = tempfile.mkdtemp(prefix="retarget_")
    temp_path = Path(temp_dir)
    source_path = temp_path / "motion_input.bvh"

    try:
        # Download BVH generated by this service so it can be uploaded to the retarget endpoint.
        download_ok = await download_file(source_bvh_url, str(source_path))
        if not download_ok:
            Log.error("Retargeting aborted: unable to fetch generated BVH from storage")
            return False

        submit_kwargs: Dict[str, Any] = {"src_file": str(source_path)}

        if RETARGET_TARGET_FILE:
            submit_kwargs["tgt_file"] = RETARGET_TARGET_FILE
        if RETARGET_SOURCE_REST_FILE:
            submit_kwargs["src_rest_file"] = RETARGET_SOURCE_REST_FILE
        if RETARGET_TARGET_REST_FILE:
            submit_kwargs["tgt_rest_file"] = RETARGET_TARGET_REST_FILE
        if preset:
            submit_kwargs["preset_name"] = preset

        response = await submit_task(RETARGET_SERVICE_URL, **submit_kwargs)
        task_id = response.get("task_id") if isinstance(response, dict) else None
        if not task_id:
            Log.error(f"Retarget service did not return task_id, response={response}")
            return False

        Log.info(f"Retarget task submitted: {task_id}")

        while True:
            status = await check_task_status(RETARGET_STATUS_URL, task_id)
            if not isinstance(status, dict):
                Log.error(f"Unexpected retarget status payload: {status}")
                return False

            if "error" in status:
                Log.error(f"Retarget status query failed: {status}")
                return False

            state = status.get("status")
            if state == "completed":
                download_url = status.get("download_url") or status.get("result_url")
                if not download_url:
                    Log.error(f"Retarget task completed without download URL: {status}")
                    return False

                if download_url.startswith("http"):
                    final_url = download_url
                else:
                    final_url = f"{BLENDER_BASE_URL.rstrip('/')}/{download_url.lstrip('/')}"

                output_path = Path(output_fbx_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                saved = await download_file(final_url, str(output_path))
                if not saved:
                    Log.error("Failed to download FBX result from retarget service")
                    return False

                Log.info(f"Retarget task {task_id} completed and saved to {output_path}")
                return True

            if state == "failed":
                err_msg = status.get("error_message") or status.get("message")
                Log.error(f"Retarget task {task_id} failed: {err_msg}")
                return False

            await asyncio.sleep(RETARGET_POLL_INTERVAL)

    except Exception as exc:
        Log.error(f"Retarget service error: {exc}")
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

async def retarget_task_async(source_bvh_url: str, target_fbx_path: str, output_fbx_path: str, preset_name: Optional[str] = None) -> bool:
    """Send BVH to external retarget service and save FBX output locally."""
    preset = preset_name or RETARGET_PRESET_NAME or None
    temp_dir = tempfile.mkdtemp(prefix="retarget_")
    temp_path = Path(temp_dir)
    source_path = temp_path / "motion_input.bvh"

    try:
        # Download BVH generated by this service so it can be uploaded to the retarget endpoint.
        download_ok = await download_file(source_bvh_url, str(source_path))
        if not download_ok:
            Log.error("Retargeting aborted: unable to fetch generated BVH from storage")
            return False

        submit_kwargs: Dict[str, Any] = {"src_file": str(source_path), "tgt_file": str(target_fbx_path)}

        if RETARGET_SOURCE_REST_FILE:
            submit_kwargs["src_rest_file"] = RETARGET_SOURCE_REST_FILE
        if RETARGET_TARGET_REST_FILE:
            submit_kwargs["tgt_rest_file"] = RETARGET_TARGET_REST_FILE
        if preset:
            submit_kwargs["preset_name"] = preset

        response = await submit_task(RETARGET_SERVICE_URL, **submit_kwargs)
        task_id = response.get("task_id") if isinstance(response, dict) else None
        if not task_id:
            Log.error(f"Retarget service did not return task_id, response={response}")
            return False

        Log.info(f"Retarget task submitted: {task_id}")

        while True:
            status = await check_task_status(RETARGET_STATUS_URL, task_id)
            if not isinstance(status, dict):
                Log.error(f"Unexpected retarget status payload: {status}")
                return False

            if "error" in status:
                Log.error(f"Retarget status query failed: {status}")
                return False

            state = status.get("status")
            if state == "completed":
                download_url = status.get("download_url") or status.get("result_url")
                if not download_url:
                    Log.error(f"Retarget task completed without download URL: {status}")
                    return False

                if download_url.startswith("http"):
                    final_url = download_url
                else:
                    final_url = f"{BLENDER_BASE_URL.rstrip('/')}/{download_url.lstrip('/')}"

                output_path = Path(output_fbx_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                saved = await download_file(final_url, str(output_path))
                if not saved:
                    Log.error("Failed to download FBX result from retarget service")
                    return False

                Log.info(f"Retarget task {task_id} completed and saved to {output_path}")
                return True

            if state == "failed":
                err_msg = status.get("error_message") or status.get("message")
                Log.error(f"Retarget task {task_id} failed: {err_msg}")
                return False

            await asyncio.sleep(RETARGET_POLL_INTERVAL)

    except Exception as exc:
        Log.error(f"Retarget service error: {exc}")
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

async def render_task_async(anim_path: str, mesh_path: str, output_video_path: str) -> bool:
    """Render task: submit animation to render service and download result."""
    try:
        response = await submit_task(RENDER_SERVICE_URL, animation=anim_path, mesh=mesh_path)
        task_id = response.get("task_id") if isinstance(response, dict) else None
        if not task_id:
            Log.error(f"Render service did not return task_id, response={response}")
            return False

        Log.info(f"Render task submitted: {task_id}")

        while True:
            status = await check_task_status(RENDER_STATUS_URL, task_id)
            if not isinstance(status, dict):
                Log.error(f"Unexpected render status payload: {status}")
                return False

            if "error" in status:
                Log.error(f"Render status query failed: {status}")
                return False

            state = status.get("status")
            if state == "completed":
                download_url = status.get("download_url") or status.get("result_url")
                if not download_url:
                    Log.error(f"Render task completed without download URL: {status}")
                    return False

                if download_url.startswith("http"):
                    final_url = download_url
                else:
                    final_url = f"{BLENDER_BASE_URL.rstrip('/')}/{download_url.lstrip('/')}"

                output_path = Path(output_video_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                saved = await download_file(final_url, str(output_path))
                if not saved:
                    Log.error("Failed to download render result from render service")
                    return False

                Log.info(f"Render task {task_id} completed and saved to {output_path}")
                return True

            if state == "failed":
                err_msg = status.get("error_message") or status.get("message")
                Log.error(f"Render task {task_id} failed: {err_msg}")
                return False

            await asyncio.sleep(RETARGET_POLL_INTERVAL)

    except Exception as exc:
        Log.error(f"Render service error: {exc}")
        return False

def _get_worker_count(queue_name: str) -> int:
    """Resolve worker count for a given queue using environment overrides."""
    env_key = f"MOTION_QUEUE_WORKERS_{queue_name.upper()}"
    default_workers = NUM_WORKERS if queue_name == "motion_generation" else QUEUE_DEFAULT_WORKERS
    try:
        return int(os.environ.get(env_key, default_workers))
    except (TypeError, ValueError):
        return default_workers


def _ensure_queue(queue_name: str) -> asyncio.Queue:
    """Ensure the async queue and worker pool for the given queue name exist."""
    if queue_name not in task_queue_dict:
        task_queue_dict[queue_name] = asyncio.Queue()
        pending_queue_map[queue_name] = deque()
        queue_workers[queue_name] = []
        Log.info(f"Created task queue: {queue_name}")

    _ensure_workers(queue_name, _get_worker_count(queue_name))
    return task_queue_dict[queue_name]


def _ensure_workers(queue_name: str, target_workers: int) -> None:
    """Spawn worker coroutines for the queue until target count reached."""
    workers = queue_workers.setdefault(queue_name, [])
    if target_workers <= 0:
        target_workers = 1

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()

    while len(workers) < target_workers:
        worker_id = len(workers)
        worker_task = loop.create_task(_worker_loop(queue_name, worker_id))
        workers.append(worker_task)
        Log.info(f"Started worker coroutine queue={queue_name}, worker_id={worker_id}")


def _update_queue_metadata(queue_name: str) -> None:
    """Recompute queue positions for tasks in the specified queue."""
    task_ids = pending_queue_map.get(queue_name)
    if task_ids is None:
        return

    queue_size = len(task_ids)
    for index, tid in enumerate(list(task_ids)):
        info = task_status.get(tid)
        if not info:
            continue
        info["queue_name"] = queue_name
        info["queue_position"] = index
        info["queue_size"] = queue_size


async def _worker_loop(queue_name: str, worker_id: int):
    """Worker coroutine: process tasks from the specified queue."""
    queue = task_queue_dict[queue_name]
    while True:
        item = await queue.get()
        task_id: Optional[str] = None
        try:
            task_id = item.get("task_id")
            if not task_id:
                Log.warning(f"Queue item missing task_id: {item}")
                continue

            with task_lock:
                pending = pending_queue_map.get(queue_name)
                if pending and task_id in pending:
                    pending.remove(task_id)
                task_info = task_status.get(task_id)
                if task_info:
                    task_info["status"] = "processing"
                    task_info["message"] = "Task processing started..."
                    task_info["queue_name"] = queue_name
                    task_info["queue_position"] = None
                    task_info["queue_size"] = len(pending) if pending is not None else 0
                    task_info["last_progress_at"] = datetime.now().isoformat()
                if pending is not None:
                    _update_queue_metadata(queue_name)

            if not task_info:
                Log.warning(f"Task ID {task_id} not found in status table, skipping")
                continue

            handler: Callable[..., Awaitable] = task_info.get("task") or generate_motion_bvh
            params = task_info.get("params", {})
            await handler(task_id=task_id, **params)
        except Exception as e:
            Log.error(f"Worker failed to process task queue={queue_name}, task_id={task_id}: {e}")
            with task_lock:
                if task_id and task_id in task_status:
                    info = task_status[task_id]
                    info["status"] = "failed"
                    info["message"] = f"Processing failed: {str(e)}"
                    info["error_message"] = str(e)
                    info["completed_at"] = datetime.now().isoformat()
        finally:
            queue.task_done()


async def _monitor_tasks_loop():
    """Periodic monitoring: update queue positions and detect processing timeouts."""
    while True:
        try:
            now = datetime.now()
            with task_lock:
                # Update queue positions
                for qname, queue_ids in pending_queue_map.items():
                    qsize = len(queue_ids)
                    for pos, tid in enumerate(list(queue_ids)):
                        if tid in task_status:
                            task_status[tid]["queue_name"] = qname
                            task_status[tid]["queue_position"] = pos
                            task_status[tid]["queue_size"] = qsize

                # Detect processing timeouts
                for tid, info in list(task_status.items()):
                    if info.get("status") == "processing":
                        last_str = info.get("last_progress_at") or info.get("created_at")
                        try:
                            last = datetime.fromisoformat(last_str) if isinstance(last_str, str) else last_str
                            if last and (now - last) > timedelta(minutes=TASK_STALE_MIN):
                                info["status"] = "failed"
                                info["message"] = "Task processing timeout"
                                info["error_message"] = "Progress not updated for a long time, marked as timeout"
                                info["completed_at"] = now.isoformat()
                        except Exception:
                            pass
        except Exception as e:
            Log.warning(f"Monitor loop exception: {e}")
        await asyncio.sleep(POLL_INTERVAL_SEC)


def _enqueue_task(queue_name: str, task_id: str):
    """Enqueue task and record queue metadata."""
    queue = _ensure_queue(queue_name)

    with task_lock:
        info = task_status[task_id]
        info["status"] = "queued"
        info["progress"] = max(info.get("progress", 0.0), 0.05)
        info["message"] = "Task queued, waiting for processing"
        info["queued_at"] = datetime.now().isoformat()
        info["last_progress_at"] = datetime.now().isoformat()
        info["queue_name"] = queue_name
        pending_queue_map.setdefault(queue_name, deque()).append(task_id)
        _update_queue_metadata(queue_name)

    payload = {"queue_name": queue_name, "task_id": task_id}
    queue.put_nowait(payload)
    Log.info(f"Task enqueued, queue={queue_name}, task_id={task_id}")


def inv_transform(data):
    global mean, std
    if isinstance(data, np.ndarray):
        return data * std[:data.shape[-1]] + mean[:data.shape[-1]]
    elif isinstance(data, torch.Tensor):
        return data * torch.from_numpy(std[:data.shape[-1]]).float().to(
            data.device
        ) + torch.from_numpy(mean[:data.shape[-1]]).float().to(data.device)
    else:
        raise TypeError("Expected data to be either np.ndarray or torch.Tensor")

def cleanup_old_tasks():
    """Clean up old completed/failed tasks to prevent memory leaks"""
    global task_status
    
    current_time = datetime.now()
    cutoff_time = current_time - timedelta(hours=MAX_TASK_RETENTION_HOURS)
    
    # Remove tasks older than retention period
    tasks_to_remove = []
    for task_id, task_info in task_status.items():
        # Parse the created_at timestamp
        try:
            created_at = datetime.fromisoformat(task_info.get("created_at", ""))
            if created_at < cutoff_time and task_info.get("status") in ["completed", "failed"]:
                tasks_to_remove.append(task_id)
        except (ValueError, TypeError):
            # Invalid timestamp, remove old task
            if task_info.get("status") in ["completed", "failed"]:
                tasks_to_remove.append(task_id)
    
    # If still too many tasks, remove oldest completed/failed ones
    if len(task_status) > MAX_TASK_COUNT:
        completed_failed_tasks = [
            (task_id, task_info) for task_id, task_info in task_status.items()
            if task_info.get("status") in ["completed", "failed"]
        ]
        
        # Sort by creation time (oldest first)
        completed_failed_tasks.sort(key=lambda x: x[1].get("created_at", ""))
        
        # Remove oldest tasks beyond the limit
        excess_count = len(task_status) - MAX_TASK_COUNT
        for i in range(min(excess_count, len(completed_failed_tasks))):
            tasks_to_remove.append(completed_failed_tasks[i][0])
    
    # Remove identified tasks
    for task_id in tasks_to_remove:
        task_status.pop(task_id, None)
    
    if tasks_to_remove:
        print(f"Cleaned up {len(tasks_to_remove)} old tasks. Current task count: {len(task_status)}")

async def rewrite_prompt_with_llm(user_text: str, user_duration: float) -> Tuple[str, float]:
    """
    Rewrite user prompt using LLM asynchronously.

    Args:
        user_text: Original user input text
        user_duration: User provided duration (<=0 means no duration provided)

    Returns:
        Tuple of (rewritten_text, duration)
    """
    global openai_client

    # Check if OpenAI client is initialized
    if openai_client is None:
        print("Warning: OpenAI client not initialized, using original text")
        return user_text, user_duration if user_duration > 0 else 8.0

    # Determine which system prompt to use
    has_valid_duration = user_duration > 0
    system_prompt = SYSTEM_PROMPT_WITH_DURATION if has_valid_duration else SYSTEM_PROMPT_WITHOUT_DURATION

    # Get model name from environment variable
    model_name = os.getenv("OPENAI_MODEL", "DeepSeek-R1-0528")

    try:
        # Run blocking OpenAI API call in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        
        def _call_llm():
            response = openai_client.chat.completions.create(
                model=model_name,
                stream=True,
                extra_body={
                    "provider": {
                        "only": [],
                        "order": [],
                        "sort": None,
                        "input_price_range": [],
                        "output_price_range": [],
                        "throughput_range": [],
                        "latency_range": [0, 3]
                    }
                },
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ]
            )

            # Collect streaming response
            full_content = ""
            reasoning_content = ""

            for chunk in response:
                if not getattr(chunk, "choices", None):
                    continue

                reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
                if reasoning:
                    reasoning_content += reasoning

                content = getattr(chunk.choices[0].delta, "content", None)
                if content:
                    full_content += content

            return full_content

        full_content = await loop.run_in_executor(None, _call_llm)
        
        print(f"LLM Response: {full_content}")

        # Parse the response
        if has_valid_duration:
            # User provided duration, just return rewritten text
            rewritten_text = full_content.strip()
            return rewritten_text, user_duration
        else:
            # Need to extract both text and duration from LLM response
            # Look for "Duration: X seconds" pattern
            duration_match = re.search(r'Duration:\s*(\d+(?:\.\d+)?)\s*seconds?', full_content, re.IGNORECASE)

            if duration_match:
                extracted_duration = float(duration_match.group(1))
                # Remove the duration line from the text
                rewritten_text = re.sub(r'\s*Duration:\s*\d+(?:\.\d+)?\s*seconds?\s*', '', full_content, flags=re.IGNORECASE).strip()
            else:
                # Fallback: use default duration if not found
                print(f"Warning: Could not extract duration from LLM response, using default 8 seconds")
                extracted_duration = 8.0
                rewritten_text = full_content.strip()

            return rewritten_text, extracted_duration

    except Exception as e:
        print(f"Error calling LLM API: {e}")
        print(f"Falling back to original text")
        # Fallback to original text if LLM fails
        return user_text, user_duration if user_duration > 0 else 8.0

def save_bvh_to_buffer(buffer, anim, names=None, frametime=1.0/30.0, order='xyz', positions=False, quater=True):
    """Save BVH to a string buffer"""
    if names is None:
        names = ["joint_" + str(i) for i in range(len(anim.parents))]
    
    channelmap_inv = {
        'x': 'Xrotation',
        'y': 'Yrotation',
        'z': 'Zrotation',
    }
    
    ordermap = {
        'x': 0,
        'y': 1,
        'z': 2,
    }
    
    def write_joint(anim, names, t, i, order='xyz', positions=False):
        joint_lines = []
        joint_lines.append("%sJOINT %s\n" % (t, names[i]))
        joint_lines.append("%s{\n" % t)
        t += '\t'
        
        joint_lines.append("%sOFFSET %f %f %f\n" % (t, anim.offsets[i, 0], anim.offsets[i, 1], anim.offsets[i, 2]))
        
        if positions:
            joint_lines.append("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" % (t,
                                                                                channelmap_inv[order[0]],
                                                                                channelmap_inv[order[1]],
                                                                                channelmap_inv[order[2]]))
        else:
            joint_lines.append("%sCHANNELS 3 %s %s %s\n" % (t,
                                                 channelmap_inv[order[0]], channelmap_inv[order[1]],
                                                 channelmap_inv[order[2]]))
        
        end_site = True
        
        for j in range(anim.shape[1]):
            if anim.parents[j] == i:
                joint_lines.extend(write_joint(anim, names, t, j, order=order, positions=positions))
                end_site = False
        
        if end_site:
            joint_lines.append("%sEnd Site\n" % t)
            joint_lines.append("%s{\n" % t)
            t += '\t'
            joint_lines.append("%sOFFSET %f %f %f\n" % (t, 0.0, 0.0, 0.0))
            t = t[:-1]
            joint_lines.append("%s}\n" % t)
        
        t = t[:-1]
        joint_lines.append("%s}\n" % t)
        
        return joint_lines
    
    t = ""
    buffer.write("%sHIERARCHY\n" % t)
    buffer.write("%sROOT %s\n" % (t, names[0]))
    buffer.write("%s{\n" % t)
    t += '\t'
    
    buffer.write("%sOFFSET %f %f %f\n" % (t, anim.offsets[0, 0], anim.offsets[0, 1], anim.offsets[0, 2]))
    buffer.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" %
            (t, channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))
    
    for i in range(anim.shape[1]):
        if anim.parents[i] == 0:
            joint_lines = write_joint(anim, names, t, i, order=order, positions=positions)
            for line in joint_lines:
                buffer.write(line)
    
    t = t[:-1]
    buffer.write("%s}\n" % t)
    
    buffer.write("MOTION\n")
    buffer.write("Frames: %i\n" % anim.shape[0])
    buffer.write("Frame Time: %f\n" % frametime)
    
    if quater:
        rots = np.degrees(qeuler_np(anim.rotations, order=order))
    else:
        rots = anim.rotations
    poss = anim.positions
    
    for i in range(anim.shape[0]):
        for j in range(anim.shape[1]):
            if positions or j == 0:
                buffer.write("%f %f %f %f %f %f " % (
                    poss[i, j, 0], poss[i, j, 1], poss[i, j, 2],
                    rots[i, j, ordermap[order[0]]], rots[i, j, ordermap[order[1]]], rots[i, j, ordermap[order[2]]]))
            else:
                buffer.write("%f %f %f " % (
                    rots[i, j, ordermap[order[0]]], rots[i, j, ordermap[order[1]]],
                    rots[i, j, ordermap[order[2]]]))
        buffer.write("\n")

def forward_kinematic_func(data):
    global cfg, skeleton
    motions = inv_transform(data)
    b, l, _ = data.shape
    global_quats, local_quats, r_pos = recover_bvh_from_rot(motions, cfg.data.joint_num, skeleton, keep_shape=False)
    _, global_pos = skeleton.fk_local_quat(local_quats, r_pos)
    global_pos = rearrange(global_pos, '(b l) j d -> b l j d', b = b)
    local_quats = rearrange(local_quats, '(b l) j d -> b l j d', b = b)
    r_pos = rearrange(r_pos, '(b l) d -> b l d', b = b)
    return global_pos, local_quats, r_pos

def load_vq_model(vq_cfg, device, mask_trans_cfg):
    vq_model = HRVQVAE(vq_cfg,
            vq_cfg.data.dim_pose,
            vq_cfg.model.down_t,
            vq_cfg.model.stride_t,
            vq_cfg.model.width,
            vq_cfg.model.depth,
            vq_cfg.model.dilation_growth_rate,
            vq_cfg.model.vq_act,
            vq_cfg.model.use_attn,
            vq_cfg.model.vq_norm)

    ckpt = torch.load(pjoin(vq_cfg.exp.root_ckpt_dir, vq_cfg.data.name, 'vq', vq_cfg.exp.name, 'model', mask_trans_cfg.vq_ckpt),
                            map_location=device, weights_only=True)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'model'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {vq_cfg.exp.name} from epoch {ckpt["ep"]}')
    vq_model.to(device)
    vq_model.eval()
    return vq_model

def load_trans_model(t2m_cfg, which_model, device, vq_cfg):
    t2m_transformer = MoMaskPlus(
        code_dim=t2m_cfg.vq.code_dim,
        latent_dim=t2m_cfg.model.latent_dim,
        ff_size=t2m_cfg.model.ff_size,
        num_layers=t2m_cfg.model.n_layers,
        num_heads=t2m_cfg.model.n_heads,
        dropout=t2m_cfg.model.dropout,
        text_dim=t2m_cfg.text_embedder.dim_embed,
        cond_drop_prob=t2m_cfg.training.cond_drop_prob,
        device=device,
        cfg=t2m_cfg,
        full_length=t2m_cfg.data.max_motion_length//4,
        scales=vq_cfg.quantizer.scales
    )
    ckpt = torch.load(pjoin(t2m_cfg.exp.root_ckpt_dir, t2m_cfg.data.name, "momask_plus", t2m_cfg.exp.name, 'model', which_model),
                      map_location=device, weights_only=True)
    if isinstance(ckpt["t2m_transformer"], collections.OrderedDict):
        t2m_transformer.load_state_dict(ckpt["t2m_transformer"])
    else:
        t2m_transformer.load_state_dict(ckpt["t2m_transformer"].state_dict())
    t2m_transformer.to(device)
    t2m_transformer.eval()
    print(f'Loading Mask Transformer {t2m_cfg.exp.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer

def load_gmr_model(device):
    gmr_cfg = load_config(pjoin("checkpoint_dir/snapmogen/gmr", "gmr_d292", 'gmr.yaml'))
    gmr_cfg.exp.checkpoint_dir = pjoin(gmr_cfg.exp.root_ckpt_dir, gmr_cfg.data.name, 'gmr', gmr_cfg.exp.name)
    gmr_cfg.exp.model_dir = pjoin(gmr_cfg.exp.checkpoint_dir, 'model')
    regressor = GlobalRegressor(dim_in=gmr_cfg.data.dim_pose-2, dim_latent=512, dim_out=2)
    ckpt = torch.load(pjoin(gmr_cfg.exp.model_dir, 'best.tar'), map_location=device)
    regressor.load_state_dict(ckpt['regressor'])
    regressor.eval()
    regressor.to(device)
    return regressor

def initialize_openai():
    """Initialize OpenAI client from environment variables"""
    global openai_client

    base_url = os.getenv("OPENAI_BASE_URL", "https://aiping.cn/api/v1")
    api_key = os.getenv("OPENAI_API_KEY", "")

    if not api_key:
        print("WARNING: OPENAI_API_KEY not set in environment variables!")
        print("Prompt rewriting will be disabled.")
        return

    openai_client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    print(f"OpenAI client initialized with base URL: {base_url}")

def initialize_minio():
    global minio_client
    # Configure MinIO settings - adjust these according to your setup
    endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    secure = os.getenv("MINIO_SECURE", "false").lower() == "true"

    minio_client = Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure
    )

    # Get bucket name from environment
    bucket_name = os.getenv("MINIO_BUCKET", "motion-bvh")

    try:
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
            print(f"Created bucket: {bucket_name}")
        print(f"Using MinIO bucket: {bucket_name}")
    except S3Error as e:
        print(f"Error with bucket {bucket_name}: {e}")
        # Don't fail startup if bucket already exists or permission issues
        pass

async def periodic_cleanup():
    """Periodic task to clean up old tasks"""
    while True:
        await asyncio.sleep(3600)  # Run every hour
        cleanup_old_tasks()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global models, cfg, skeleton, mean, std, retargeter, openai_client, minio_client

    print("Loading models...")
    
    # Load configuration
    cfg = load_config("./config/eval_momaskplus.yaml")
    fixseed(cfg.seed)
    
    if cfg.device != 'cpu':
        torch.cuda.set_device(cfg.device)
    device = torch.device(cfg.device)
    torch.autograd.set_detect_anomaly(True)
    
    # Setup directories
    cfg.checkpoint_dir = pjoin(cfg.root_ckpt_dir, cfg.data.name, 'momask_plus', cfg.mask_trans_name)
    cfg.model_dir = pjoin(cfg.checkpoint_dir, 'model')
    meta_dir = pjoin(cfg.data.root_dir, 'meta_data')
    
    # Load configurations
    mask_trans_cfg = load_config(pjoin(cfg.root_ckpt_dir, cfg.data.name, 'momask_plus', cfg.mask_trans_name, 'train_momaskplus.yaml'))
    vq_cfg = load_config(pjoin(cfg.root_ckpt_dir, cfg.data.name, 'vq', mask_trans_cfg.vq_name, 'residual_vqvae.yaml'))
    mask_trans_cfg.vq = vq_cfg.quantizer
    
    # Load models
    models['vq_model'] = load_vq_model(vq_cfg, device, mask_trans_cfg)
    models['gmr_model'] = load_gmr_model(device)
    models['t2m_transformer'] = load_trans_model(mask_trans_cfg, cfg.which_epoch, device, vq_cfg)
    
    # Load mean and std
    mean = np.load(pjoin(meta_dir, 'mean.npy'))
    std = np.load(pjoin(meta_dir, 'std.npy'))
    
    # Load skeleton
    template_anim = bvh_io.load(pjoin(cfg.data.root_dir, 'renamed_bvhs', 'm_ep2_00086.bvh'))
    skeleton = Skeleton(template_anim.offsets, template_anim.parents, device=device)
    models['template_anim'] = template_anim
    
    # Initialize retargeter
    retargeter = RestPoseRetargeter()

    # Initialize OpenAI client
    initialize_openai()

    # Initialize MinIO
    initialize_minio()

    print("Models loaded successfully!")
    
    # Initialize motion generation queue
    _ensure_queue("motion_generation")
    
    # Start periodic cleanup task
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    # Start task monitoring loop
    monitor_task = asyncio.create_task(_monitor_tasks_loop())
    
    yield
    
    # Shutdown
    print("Shutting down...")
    cleanup_task.cancel()
    monitor_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass

app = FastAPI(title="Motion Generation Service", lifespan=lifespan)

async def generate_motion_bvh(task_id: str, text: str, duration: float):
    """Background task to generate motion and upload to MinIO"""
    global models, cfg, skeleton, mean, std, retargeter, minio_client, task_status
    await asyncio.sleep(0.01)  # Yield control to event loop
    try:
        # Update status to processing
        task_status[task_id]["status"] = "processing"
        task_status[task_id]["progress"] = 0.1
        task_status[task_id]["message"] = "Starting prompt rewriting..."

        rewritten_text, final_duration = await rewrite_prompt_with_llm(text, duration)
        print(f"Rewritten text: {rewritten_text}, final duration: {final_duration}")

        # Update status to processing
        task_status[task_id]["status"] = "processing"
        task_status[task_id]["progress"] = 0.3
        task_status[task_id]["message"] = "Starting motion generation..."
        
        device = torch.device(cfg.device)
        
        # Convert duration to frames (30 FPS)
        frames = int(final_duration * 30)
        # Ensure frames is within valid range and divisible by 4
        frames = max(7*30, min(frames, 10*30))  # 7-10 seconds range
        frames = (frames // 4) * 4  # Make divisible by 4
        
        m_length = torch.tensor([frames], device=device).long()
        
        task_status[task_id]["progress"] = 0.3
        task_status[task_id]["message"] = "Generating motion tokens..."
        
        # Generate motion
        mids = models['t2m_transformer'].generate([rewritten_text], m_length//4, cfg.time_steps[0], cfg.cond_scales[0], temperature=1)
        
        task_status[task_id]["progress"] = 0.5
        task_status[task_id]["message"] = "Decoding motion..."
        
        pred_motions = models['vq_model'].forward_decoder(mids, m_length)
        gen_global_pos, gen_local_quat, gen_r_pos = forward_kinematic_func(pred_motions)
        
        task_status[task_id]["progress"] = 0.7
        task_status[task_id]["message"] = "Creating animation..."
        
        # Create animation
        template_anim = models['template_anim']
        gen_anim = Animation(
            gen_local_quat[0, :m_length[0]].detach().cpu().numpy(), 
            repeat(gen_r_pos[0, :m_length[0]].detach().cpu().numpy(), 'i j -> i k j', k=len(template_anim)),
            template_anim.orients, 
            template_anim.offsets, 
            template_anim.parents, 
            template_anim.names, 
            template_anim.frametime
        )
        
        # Process through GMR
        feats = process_bvh_motion(None, 30, 30, 0.11, shift_one_frame=True, animation=gen_anim)
        feats = (feats - mean) / std
        feats = torch.from_numpy(feats).unsqueeze(0).float().to(device)
        gmr_input = torch.cat([feats[..., 0:1], feats[..., 3:cfg.data.dim_pose-4]], dim=-1)
        gmr_output = models['gmr_model'](gmr_input)
        rec_feats = torch.cat([feats[..., 0:1], gmr_output, feats[..., 3:]], dim=-1)
        
        single_gen_global_pos, single_gen_local_quat, single_gen_r_pos = forward_kinematic_func(rec_feats)
        
        task_status[task_id]["progress"] = 0.9
        task_status[task_id]["message"] = "Generating BVH file..."
        
        # Create final animation
        new_anim = Animation(
            single_gen_local_quat[0].detach().cpu().numpy(), 
            repeat(single_gen_r_pos[0].detach().cpu().numpy(), 'i j -> i k j', k=len(template_anim)),
            template_anim.orients, 
            template_anim.offsets, 
            template_anim.parents, 
            template_anim.names, 
            template_anim.frametime
        )
        
        # Generate BVH content
        retargeted_anim = retargeter.rest_pose_retarget(new_anim, tgt_rest='T')
        
        # Save BVH to memory buffer
        bvh_buffer = io.StringIO()
        save_bvh_to_buffer(bvh_buffer, retargeted_anim, names=new_anim.names, 
                          frametime=new_anim.frametime, order='xyz', quater=True)
        bvh_content = bvh_buffer.getvalue()
        
        # Upload to MinIO
        bucket_name = os.getenv("MINIO_BUCKET", "motion-bvh")
        object_name = f"motion_{task_id}.bvh"
        bvh_bytes = bvh_content.encode('utf-8')
        
        minio_client.put_object(
            bucket_name=bucket_name,
            object_name=object_name,
            data=io.BytesIO(bvh_bytes),
            length=len(bvh_bytes),
            content_type="text/plain"
        )
        
        # Generate presigned URL (valid for 24 hours)
        download_generated_url = minio_client.presigned_get_object(bucket_name, object_name, expires=timedelta(hours=24))
        
        # Update task status
        task_status[task_id]["status"] = "retargeting"
        task_status[task_id]["progress"] = 0.8
        task_status[task_id]["message"] = "Motion generation completed successfully"
        task_status[task_id]["completed_at"] = datetime.now().isoformat()
        task_status[task_id]["download_url"] = download_generated_url

        temp_dir = f"/tmp/snap_{task_id}"
        os.makedirs(temp_dir, exist_ok=True)

        # retarget to Momax
        fbx_path = f"{temp_dir}/motion_{task_id}.fbx"
        retarget_success = await retarget_task_async(download_generated_url, RETARGET_TARGET_FILE, output_fbx_path=fbx_path, preset_name=RETARGET_PRESET_NAME)

        # upload retargeted fbx if successful
        if retarget_success:
            # Upload FBX to MinIO
            fbx_object_name = f"motion_{task_id}.fbx"
            with open(f"motion_{task_id}.fbx", "rb") as fbx_file:
                fbx_data = fbx_file.read()
                minio_client.put_object(
                    bucket_name=bucket_name,
                    object_name=fbx_object_name,
                    data=io.BytesIO(fbx_data),
                    length=len(fbx_data),
                    content_type="application/octet-stream"
                )
            
            # Generate presigned URL for FBX (valid for 24 hours)
            download_fbx_url = minio_client.presigned_get_object(bucket_name, fbx_object_name, expires=timedelta(hours=24))
            
            # Update task status with FBX download URL
            task_status[task_id]["status"] = "rendering"
            task_status[task_id]["progress"] = 0.9
            task_status[task_id]["message"] = "Motion is rendering"
            task_status[task_id]["completed_at"] = datetime.now().isoformat()
            task_status[task_id]["result_url"] = {"retargeted": download_fbx_url}
            task_status[task_id]["download_url"] = download_fbx_url
            
            task_status[task_id]["message"] = "Motion generation and retargeting completed successfully"
        else:
            task_status[task_id]["status"] = "failed"
            Log.warning("Retargeting to SMPL-X failed; BVH output remains available")
            raise Exception("Retargeting to SMPL-X failed")

        video_path = f"{temp_dir}/motion_{task_id}.mp4"
        render_success = await render_task_async(fbx_path, RETARGET_TARGET_FILE, output_video_path=video_path)
        if render_success:
            # Upload video to MinIO
            video_object_name = f"motion_{task_id}.mp4"
            with open(video_path, "rb") as video_file:
                video_data = video_file.read()
                minio_client.put_object(
                    bucket_name=bucket_name,
                    object_name=video_object_name,
                    data=io.BytesIO(video_data),
                    length=len(video_data),
                    content_type="video/mp4"
                )
            
            # Generate presigned URL for video (valid for 24 hours)
            download_video_url = minio_client.presigned_get_object(bucket_name, video_object_name, expires=timedelta(hours=24))
            
            # Update task status with video download URL
            task_status[task_id]["status"] = "completed"
            task_status[task_id]["progress"] = 1.0
            task_status[task_id]["message"] = "Motion generation, retargeting, and rendering completed successfully"
            task_status[task_id]["completed_at"] = datetime.now().isoformat()
            task_status[task_id]["result_url"]["video"] = download_video_url
        else:
            task_status[task_id]["status"] = "failed"
            Log.warning("Rendering video failed; BVH and FBX outputs remain available")
            raise Exception("Rendering video failed")
        
        # Clean up old tasks to prevent memory leaks
        cleanup_old_tasks()
        
    except Exception as e:
        task_status[task_id]["status"] = "failed"
        task_status[task_id]["message"] = f"Error: {str(e)}"
        task_status[task_id]["completed_at"] = datetime.now().isoformat()
        print(f"Error in task {task_id}: {e}")
        
        # Clean up old tasks even on failure
        cleanup_old_tasks()
        raise e

@app.post("/upload", response_model=UploadResponse)
async def upload_motion_request(request: MotionRequest, background_tasks: BackgroundTasks):
    """Submit a motion generation request"""

    # Validate input
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text description cannot be empty")

    if request.duration > 12:
        raise HTTPException(status_code=400, detail="Duration must be no more than 12 seconds")

    # Rewrite prompt with LLM
    print(f"Original text: {request.text}, duration: {request.duration}")

    # Generate unique task ID
    task_id = str(uuid.uuid4())

    # Initialize task status
    with task_lock:
        task_status[task_id] = {
            "task_id": task_id,
            "task": generate_motion_bvh,
            "params": {"text": request.text, "duration": request.duration},
            "status": "pending",
            "progress": 0.0,
            "message": "Task created",
            "created_at": datetime.now().isoformat(),
            "text": request.text,
            "original_text": request.text,
            "duration": request.duration
        }

    # Enqueue task instead of using background_tasks
    _enqueue_task(queue_name="motion_generation", task_id=task_id)

    # Return task status with queue info
    with task_lock:
        queue_name = task_status[task_id].get("queue_name", "motion_generation")
        queue_snapshot = list(pending_queue_map.get(queue_name, deque()))
        pos = queue_snapshot.index(task_id) if task_id in queue_snapshot else None

    return UploadResponse(
        task_id=task_id,
        status="queued",
        message=f"Motion generation task submitted and queued (position: {pos}, queue size: {len(queue_snapshot)})"
    )

@app.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get the status of a motion generation task"""
    
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = task_status[task_id]
    return TaskStatusResponse(**task)

@app.get("/download/{task_id}")
async def get_download_url(task_id: str):
    """Get the download URL for a completed motion generation task"""
    
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = task_status[task_id]
    
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Task is not completed. Current status: {task['status']}")
    
    if "download_url" not in task:
        raise HTTPException(status_code=500, detail="Download URL not available")
    
    return {"download_url": task["download_url"]}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/queue")
async def get_queue_state():
    """Get current queue state"""
    with task_lock:
        queues = {
            name: {
                "queue_size": len(ids),
                "task_ids": list(ids),
            }
            for name, ids in pending_queue_map.items()
        }
        total_size = sum(len(ids) for ids in pending_queue_map.values())
        return {
            "total_queue_size": total_size,
            "queues": queues,
        }

@app.get("/tasks")
async def list_tasks():
    """List all tasks with their status"""
    with task_lock:
        tasks = []
        for task_id, task_info in task_status.items():
            # Calculate task age
            created_at_str = task_info.get("created_at", "")
            try:
                created_at = datetime.fromisoformat(created_at_str) if created_at_str else datetime.now()
                age = datetime.now() - created_at
                age_minutes = age.total_seconds() / 60
            except:
                age_minutes = 0
            
            item = {
                "task_id": task_id,
                "status": task_info.get("status"),
                "progress": task_info.get("progress"),
                "message": task_info.get("message"),
                "age_minutes": round(age_minutes, 1),
                "created_at": created_at_str,
            }
            
            # Add queue info if available
            if "queue_position" in task_info:
                item["queue_position"] = task_info.get("queue_position")
                item["queue_size"] = task_info.get("queue_size")
            if task_info.get("queue_name"):
                item["queue_name"] = task_info.get("queue_name")
            
            tasks.append(item)
        
        return {
            "total_tasks": len(tasks),
            "tasks": tasks
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)