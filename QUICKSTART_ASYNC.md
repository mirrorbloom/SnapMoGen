# Quick Start Guide - Async Motion Service

## Prerequisites
1. Blender retarget/render service running at `http://localhost:8001`
2. MinIO instance running at `localhost:9000`
3. Asset file `./assets/MomaxBot.fbx` exists
4. Python environment with all dependencies installed

## Configuration

### 1. Check .env.dev
Ensure `.env.dev` exists with proper configuration:
```bash
cat .env.dev
```

### 2. Override Defaults (Optional)
Export environment variables to override defaults:
```bash
export BLENDER_BASE_URL=http://your-blender-service:8001
export MINIO_ENDPOINT=your-minio:9000
export MINIO_ACCESS_KEY=your-key
export MINIO_SECRET_KEY=your-secret
export MOTION_NUM_WORKERS=2  # Increase for parallel processing
```

## Starting the Service

```bash
# Start the motion service
python motion_service.py
```

Expected output:
```
Loaded environment variables from /path/to/.env.dev
Loading models...
Loading VQ Model ...
Loading Mask Transformer ...
Models loaded successfully!
Created task queue: motion_generation
Started worker coroutine queue=motion_generation, worker_id=0
INFO:     Started server process [12345]
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Testing the Service

### 1. Submit a Motion Generation Request
```bash
curl -X POST http://localhost:8000/upload \
  -H "Content-Type: application/json" \
  -d '{
    "text": "A person walking forward slowly",
    "duration": 8.0
  }'
```

Response:
```json
{
  "task_id": "abc123",
  "status": "queued",
  "message": "Motion generation task submitted and queued (position: 0, queue size: 1)"
}
```

### 2. Check Task Status
```bash
curl http://localhost:8000/status/abc123
```

Response (in progress):
```json
{
  "task_id": "abc123",
  "status": "processing",
  "progress": 0.6,
  "message": "Retargeting completed, starting render...",
  "created_at": "2025-01-01T10:00:00",
  "queue_name": "motion_generation",
  "last_progress_at": "2025-01-01T10:01:23",
  "result_urls": {
    "bvh": "http://minio:9000/motion-bvh/motion_abc123.bvh?...",
    "fbx": "http://minio:9000/motion-bvh/motion_abc123.fbx?..."
  }
}
```

Response (completed):
```json
{
  "task_id": "abc123",
  "status": "completed",
  "progress": 1.0,
  "message": "Motion generation, retargeting, and rendering completed successfully",
  "created_at": "2025-01-01T10:00:00",
  "completed_at": "2025-01-01T10:02:45",
  "download_url": "http://minio:9000/motion-bvh/motion_abc123.fbx?...",
  "result_urls": {
    "bvh": "http://minio:9000/motion-bvh/motion_abc123.bvh?...",
    "fbx": "http://minio:9000/motion-bvh/motion_abc123.fbx?...",
    "rendered_video": "http://minio:9000/motion-bvh/rendered_abc123.mp4?..."
  }
}
```

### 3. Check Queue Status
```bash
curl http://localhost:8000/queue
```

Response:
```json
{
  "total_queue_size": 2,
  "queues": {
    "motion_generation": {
      "queue_size": 2,
      "task_ids": ["abc123", "def456"]
    }
  }
}
```

### 4. List All Tasks
```bash
curl http://localhost:8000/tasks
```

Response:
```json
{
  "total_tasks": 3,
  "tasks": [
    {
      "task_id": "abc123",
      "status": "completed",
      "progress": 1.0,
      "message": "Completed",
      "age_minutes": 5.2,
      "created_at": "2025-01-01T10:00:00"
    },
    {
      "task_id": "def456",
      "status": "processing",
      "progress": 0.4,
      "message": "Creating animation...",
      "age_minutes": 2.1,
      "created_at": "2025-01-01T10:03:00",
      "queue_name": "motion_generation"
    },
    {
      "task_id": "ghi789",
      "status": "queued",
      "progress": 0.05,
      "message": "Task queued, waiting for processing",
      "age_minutes": 0.5,
      "created_at": "2025-01-01T10:04:30",
      "queue_name": "motion_generation",
      "queue_position": 0,
      "queue_size": 1
    }
  ]
}
```

### 5. Download Results
```bash
# Download BVH
curl -o motion.bvh "$(curl -s http://localhost:8000/status/abc123 | jq -r '.result_urls.bvh')"

# Download FBX
curl -o motion.fbx "$(curl -s http://localhost:8000/status/abc123 | jq -r '.result_urls.fbx')"

# Download rendered video
curl -o motion.mp4 "$(curl -s http://localhost:8000/status/abc123 | jq -r '.result_urls.rendered_video')"
```

## Monitoring

### Health Check
```bash
curl http://localhost:8000/health
```

### View Logs
```bash
# Service logs show:
# - Queue initialization
# - Task enqueuing
# - Worker processing
# - Progress updates
# - External service calls
# - Errors and warnings
```

## Troubleshooting

### Task Stuck in "queued"
- Check worker count: `MOTION_NUM_WORKERS` env var
- Check queue status: `GET /queue`
- Restart service to respawn workers

### Task Stuck at "retargeting"
- Verify Blender service is running
- Check `RETARGET_SERVICE_URL` and `RETARGET_STATUS_URL`
- Check Blender service logs

### Task Stuck at "rendering"
- Verify render service is running
- Check `RENDER_SERVICE_URL` and `RENDER_STATUS_URL`
- Verify `MOMAX_BOT_PATH` points to valid FBX file

### Task Timeout
- Increase `MOTION_TASK_STALE_MIN` (default 60 minutes)
- Check if worker crashed (service logs)

### MinIO Upload Fails
- Verify MinIO is accessible
- Check credentials in `.env.dev`
- Ensure bucket exists (service creates it automatically)

## Advanced Configuration

### Increase Concurrency
```bash
# Process 3 tasks simultaneously
export MOTION_NUM_WORKERS=3
```

### Adjust Polling Intervals
```bash
# Check retarget/render status every 1 second
export RETARGET_POLL_INTERVAL=1.0

# Monitor tasks every 2 seconds
export MOTION_POLL_INTERVAL_SEC=2
```

### Custom Asset Path
```bash
export ASSET_PATH=/path/to/custom/assets
export MOMAX_BOT_PATH=/path/to/custom/MomaxBot.fbx
```

## API Documentation

Full API documentation available at:
```
http://localhost:8000/docs
```

Or with ReDoc:
```
http://localhost:8000/redoc
```
