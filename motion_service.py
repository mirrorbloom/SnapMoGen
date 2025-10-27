import warnings
warnings.filterwarnings("ignore")

import os
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from os.path import join as pjoin
import re

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from minio import Minio
from minio.error import S3Error
import io
from contextlib import asynccontextmanager
from openai import OpenAI

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

class UploadResponse(BaseModel):
    task_id: str
    status: str
    message: str

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

def rewrite_prompt_with_llm(user_text: str, user_duration: float) -> Tuple[str, float]:
    """
    Rewrite user prompt using LLM.

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
        # Call LLM API with streaming
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
    
    # Start periodic cleanup task
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    yield
    
    # Shutdown
    print("Shutting down...")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

app = FastAPI(title="Motion Generation Service", lifespan=lifespan)

async def generate_motion_bvh(task_id: str, text: str, duration: float):
    """Background task to generate motion and upload to MinIO"""
    global models, cfg, skeleton, mean, std, retargeter, minio_client, task_status
    
    try:
        # Update status to processing
        task_status[task_id]["status"] = "processing"
        task_status[task_id]["progress"] = 0.1
        task_status[task_id]["message"] = "Starting motion generation..."
        
        device = torch.device(cfg.device)
        
        # Convert duration to frames (30 FPS)
        frames = int(duration * 30)
        # Ensure frames is within valid range and divisible by 4
        frames = max(7*30, min(frames, 10*30))  # 7-10 seconds range
        frames = (frames // 4) * 4  # Make divisible by 4
        
        m_length = torch.tensor([frames], device=device).long()
        
        task_status[task_id]["progress"] = 0.3
        task_status[task_id]["message"] = "Generating motion tokens..."
        
        # Generate motion
        mids = models['t2m_transformer'].generate([text], m_length//4, cfg.time_steps[0], cfg.cond_scales[0], temperature=1)
        
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
        retargeted_anim = retargeter.rest_pose_retarget(new_anim, tgt_rest='A')
        
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
        download_url = minio_client.presigned_get_object(bucket_name, object_name, expires=timedelta(hours=24))
        
        # Update task status
        task_status[task_id]["status"] = "completed"
        task_status[task_id]["progress"] = 1.0
        task_status[task_id]["message"] = "Motion generation completed successfully"
        task_status[task_id]["completed_at"] = datetime.now().isoformat()
        task_status[task_id]["download_url"] = download_url
        
        # Clean up old tasks to prevent memory leaks
        cleanup_old_tasks()
        
    except Exception as e:
        task_status[task_id]["status"] = "failed"
        task_status[task_id]["message"] = f"Error: {str(e)}"
        task_status[task_id]["completed_at"] = datetime.now().isoformat()
        print(f"Error in task {task_id}: {e}")
        
        # Clean up old tasks even on failure
        cleanup_old_tasks()

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
    rewritten_text, final_duration = rewrite_prompt_with_llm(request.text, request.duration)
    print(f"Rewritten text: {rewritten_text}, final duration: {final_duration}")

    # Generate unique task ID
    task_id = str(uuid.uuid4())

    # Initialize task status
    task_status[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "progress": 0.0,
        "message": "Task queued for processing",
        "created_at": datetime.now().isoformat(),
        "text": rewritten_text,
        "original_text": request.text,
        "duration": final_duration
    }

    # Start background task with rewritten text and final duration
    background_tasks.add_task(generate_motion_bvh, task_id, rewritten_text, final_duration)

    return UploadResponse(
        task_id=task_id,
        status="pending",
        message="Motion generation task submitted successfully"
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)