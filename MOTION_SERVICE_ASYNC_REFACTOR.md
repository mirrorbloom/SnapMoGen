# Motion Service Async Refactor

## Summary
Refactored `motion_service.py` to use async queue-based task processing similar to the GVHMR script pattern, with integrated retargeting and rendering pipeline.

## Key Changes

### 1. Added Async Queue Infrastructure
- **Task queues**: `task_queue_dict` for managing async queues per queue name
- **Pending queue tracking**: `pending_queue_map` for queue position metadata
- **Worker pools**: `queue_workers` for managing worker coroutines
- **Queue functions**:
  - `_ensure_queue()`: Creates queue and spawns workers
  - `_ensure_workers()`: Manages worker coroutine lifecycle
  - `_worker_loop()`: Processes tasks from queue
  - `_enqueue_task()`: Adds tasks to queue with metadata
  - `_update_queue_metadata()`: Maintains queue position info
  - `_monitor_tasks_loop()`: Periodic monitoring for timeouts

### 2. Enhanced Task Processing Pipeline
The `generate_motion_bvh()` function now includes:
1. **Motion Generation** (0-50% progress)
   - Generate motion tokens from text
   - Decode motion through VQ-VAE
   - Apply GMR refinement
   - Create BVH file

2. **Retargeting** (50-70% progress)
   - Upload BVH to MinIO
   - Call external retarget service
   - Poll for completion
   - Download and upload FBX result

3. **Rendering** (70-100% progress)
   - Submit FBX + mesh to render service
   - Poll for render completion
   - Download and upload rendered video

### 3. External Service Integration
Added async helpers for external Blender services:
- `retarget_to_smplx_async()`: Retarget BVH to SMPL-X FBX
- `render_task_async()`: Render FBX animation with mesh

### 4. Configuration Management
Updated `.env.dev` with:
- Render service URLs
- Asset paths (MomaxBot.fbx)
- Queue worker configuration
- Task timeout settings

### 5. Enhanced Task Status
Extended `TaskStatusResponse` with queue metadata:
- `queue_name`: Which queue is processing the task
- `queue_position`: Position in queue (0-indexed)
- `queue_size`: Total tasks in queue
- `queued_at`: When task was enqueued
- `last_progress_at`: Last progress update timestamp
- `result_urls`: Dict of all result file URLs (BVH, FBX, video)

### 6. New API Endpoints
- `GET /queue`: View current queue state
- `GET /tasks`: List all tasks with queue metadata

## Environment Variables

```bash
# Blender/Retarget Services
BLENDER_BASE_URL=http://localhost:8001
RETARGET_SERVICE_URL=http://localhost:8001/retarget
RETARGET_STATUS_URL=http://localhost:8001/retarget_status
RENDER_SERVICE_URL=http://localhost:8001/render_animation
RENDER_STATUS_URL=http://localhost:8001/render_status
RETARGET_POLL_INTERVAL=2.0

# Assets
ASSET_PATH=./assets
MOMAX_BOT_PATH=./assets/MomaxBot.fbx

# Queue Configuration
MOTION_QUEUE_DEFAULT_WORKERS=1
MOTION_NUM_WORKERS=1
MOTION_POLL_INTERVAL_SEC=5
MOTION_TASK_STALE_MIN=60

# Storage
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=motion-bvh
MINIO_SECURE=false
```

## Workflow

### Previous Flow
1. POST /upload → BackgroundTasks → generate_motion_bvh
2. BVH uploaded to MinIO
3. Task completed

### New Flow
1. POST /upload → Task created → Enqueued to "motion_generation" queue
2. Worker picks up task → generate_motion_bvh
3. Generate BVH → Upload to MinIO (result_urls["bvh"])
4. Call retarget service → Download FBX → Upload to MinIO (result_urls["fbx"])
5. Call render service → Download video → Upload to MinIO (result_urls["rendered_video"])
6. Task marked completed with all URLs available

## Graceful Degradation
- If retargeting fails: Task completes with BVH only
- If rendering fails: Task completes with BVH + FBX only
- All steps update progress and last_progress_at for timeout monitoring

## Testing Checklist
- [ ] Verify .env.dev loads correctly
- [ ] Check queue initialization on startup
- [ ] Test motion generation with valid prompt
- [ ] Verify retarget service integration
- [ ] Verify render service integration
- [ ] Test queue status endpoints
- [ ] Test task timeout handling
- [ ] Verify result URLs are accessible
- [ ] Test with multiple concurrent requests

## Migration Notes
- Old clients using `download_url` will still work (points to final FBX or BVH)
- New clients can use `result_urls` dict to access specific file types
- Queue-based processing allows better load management and progress tracking
- External service failures won't crash the main service
