# Motion Generation Service API Documentation

## 概述

Motion Generation Service 是一个基于文本生成人体动作的 API 服务。它使用深度学习模型从文本描述生成 3D 人体动作数据，并支持自动重定向到标准骨骼格式和渲染预览视频。

**Base URL**: `http://localhost:8000`

**服务特性**:
- 文本到动作生成（Text-to-Motion）
- 自动 BVH → FBX 格式转换
- 动作重定向到 SMPL-X 骨骼
- 自动渲染预览视频
- 异步任务队列处理
- 支持 LLM 提示词重写优化

---

## 认证

当前版本不需要认证。

---

## API 端点

### 1. 提交动作生成任务

创建一个新的动作生成任务。

**端点**: `POST /upload`

**请求体**:
```json
{
  "text": "A person walking forward slowly",
  "duration": 8.0
}
```

**参数说明**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `text` | string | 是 | 动作描述文本（中文或英文） |
| `duration` | float | 是 | 动作时长（秒），范围 2-12 秒 |

**响应示例**:
```json
{
  "task_id": "abc123def456",
  "status": "queued",
  "message": "Motion generation task submitted and queued (position: 0, queue size: 1)"
}
```

**状态码**:
- `200 OK` - 任务创建成功
- `400 Bad Request` - 参数错误（文本为空或时长超限）
- `500 Internal Server Error` - 服务器内部错误

**示例请求**:
```bash
curl -X POST http://localhost:8000/upload \
  -H "Content-Type: application/json" \
  -d '{
    "text": "一个人慢慢向前走",
    "duration": 8.0
  }'
```

---

### 2. 查询任务状态

查询指定任务的处理状态和进度。

**端点**: `GET /status/{task_id}`

**路径参数**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `task_id` | string | 任务 ID（从 POST /upload 返回） |

**响应示例**:

**处理中**:
```json
{
  "task_id": "abc123def456",
  "status": "processing",
  "progress": 0.6,
  "message": "Retargeting completed, starting render...",
  "created_at": "2025-10-29T10:00:00",
  "queue_name": "motion_generation",
  "last_progress_at": "2025-10-29T10:01:23",
  "result_urls": {
    "bvh": "http://minio:9000/motion-bvh/motion_abc123.bvh?X-Amz-...",
    "fbx": "http://minio:9000/motion-bvh/motion_abc123.fbx?X-Amz-..."
  }
}
```

**已完成**:
```json
{
  "task_id": "abc123def456",
  "status": "completed",
  "progress": 1.0,
  "message": "Motion generation, retargeting, and rendering completed successfully",
  "created_at": "2025-10-29T10:00:00",
  "completed_at": "2025-10-29T10:02:45",
  "download_url": "http://minio:9000/motion-bvh/motion_abc123.fbx?X-Amz-...",
  "result_urls": {
    "bvh": "http://minio:9000/motion-bvh/motion_abc123.bvh?X-Amz-...",
    "fbx": "http://minio:9000/motion-bvh/motion_abc123.fbx?X-Amz-...",
    "rendered_video": "http://minio:9000/motion-bvh/rendered_abc123.mp4?X-Amz-..."
  }
}
```

**任务状态说明**:

| 状态 | 说明 |
|------|------|
| `queued` | 任务已排队，等待处理 |
| `processing` | 任务正在处理中 |
| `completed` | 任务已完成 |
| `failed` | 任务失败 |

**进度范围**:
- `0.0 - 0.2`: 生成动作 tokens
- `0.2 - 0.5`: 解码动作数据
- `0.5 - 0.6`: 生成 BVH 文件
- `0.6 - 0.7`: 重定向到 FBX
- `0.7 - 1.0`: 渲染预览视频

**状态码**:
- `200 OK` - 成功返回任务状态
- `404 Not Found` - 任务 ID 不存在

**示例请求**:
```bash
curl http://localhost:8000/status/abc123def456
```

---

### 3. 获取下载链接

获取已完成任务的结果文件下载链接（兼容旧版 API）。

**端点**: `GET /download/{task_id}`

**路径参数**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `task_id` | string | 任务 ID |

**响应示例**:
```json
{
  "download_url": "http://minio:9000/motion-bvh/motion_abc123.fbx?X-Amz-..."
}
```

**状态码**:
- `200 OK` - 成功返回下载链接
- `400 Bad Request` - 任务尚未完成
- `404 Not Found` - 任务 ID 不存在
- `500 Internal Server Error` - 下载链接不可用

**示例请求**:
```bash
curl http://localhost:8000/download/abc123def456
```

**注意**: 推荐使用 `GET /status/{task_id}` 返回的 `result_urls` 字段获取不同格式的文件链接。

---

### 4. 查看队列状态

查看当前任务队列的状态。

**端点**: `GET /queue`

**响应示例**:
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

**字段说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `total_queue_size` | integer | 所有队列的总任务数 |
| `queues` | object | 各队列详情 |
| `queues.{name}.queue_size` | integer | 该队列的任务数 |
| `queues.{name}.task_ids` | array | 队列中的任务 ID 列表 |

**状态码**:
- `200 OK` - 成功返回队列状态

**示例请求**:
```bash
curl http://localhost:8000/queue
```

---

### 5. 列出所有任务

列出所有任务及其状态信息。

**端点**: `GET /tasks`

**响应示例**:
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
      "created_at": "2025-10-29T10:00:00"
    },
    {
      "task_id": "def456",
      "status": "processing",
      "progress": 0.4,
      "message": "Creating animation...",
      "age_minutes": 2.1,
      "created_at": "2025-10-29T10:03:00",
      "queue_name": "motion_generation"
    },
    {
      "task_id": "ghi789",
      "status": "queued",
      "progress": 0.05,
      "message": "Task queued, waiting for processing",
      "age_minutes": 0.5,
      "created_at": "2025-10-29T10:04:30",
      "queue_name": "motion_generation",
      "queue_position": 0,
      "queue_size": 1
    }
  ]
}
```

**字段说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `total_tasks` | integer | 任务总数 |
| `tasks` | array | 任务列表 |
| `tasks[].task_id` | string | 任务 ID |
| `tasks[].status` | string | 任务状态 |
| `tasks[].progress` | float | 进度（0.0-1.0） |
| `tasks[].message` | string | 状态消息 |
| `tasks[].age_minutes` | float | 任务创建后经过的分钟数 |
| `tasks[].created_at` | string | 创建时间（ISO 8601） |
| `tasks[].queue_name` | string | 队列名称（如果在队列中） |
| `tasks[].queue_position` | integer | 队列位置（如果在队列中） |
| `tasks[].queue_size` | integer | 队列大小（如果在队列中） |

**状态码**:
- `200 OK` - 成功返回任务列表

**示例请求**:
```bash
curl http://localhost:8010/tasks
```

---

### 6. 健康检查

检查服务是否正常运行。

**端点**: `GET /health`

**响应示例**:
```json
{
  "status": "healthy",
  "timestamp": "2025-10-29T10:30:00"
}
```

**状态码**:
- `200 OK` - 服务正常运行

**示例请求**:
```bash
curl http://localhost:8000/health
```

---

## 数据模型

### MotionRequest

请求模型（POST /upload）

```typescript
{
  text: string;      // 动作描述，非空
  duration: number;  // 时长（秒），范围 0-12
}
```

### TaskStatusResponse

任务状态响应模型

```typescript
{
  task_id: string;
  status: "queued" | "processing" | "completed" | "failed";
  progress?: number;              // 0.0-1.0
  message?: string;
  created_at: string;             // ISO 8601 格式
  completed_at?: string;          // ISO 8601 格式
  download_url?: string;          // 主要结果文件 URL（FBX 或 BVH）
  result_urls?: {
    bvh?: string;                 // BVH 文件 URL
    fbx?: string;                 // FBX 文件 URL
    rendered_video?: string;      // 渲染视频 URL
  };
  error_message?: string;         // 错误消息（status=failed 时）
  queue_position?: number;        // 队列位置
  queue_size?: number;            // 队列大小
  queue_name?: string;            // 队列名称
  queued_at?: string;             // 入队时间
  last_progress_at?: string;      // 最后更新时间
}
```

### UploadResponse

上传响应模型

```typescript
{
  task_id: string;
  status: string;
  message: string;
}
```

---

## 使用流程

### 基本流程

1. **提交任务**
   ```bash
   curl -X POST http://localhost:8000/upload \
     -H "Content-Type: application/json" \
     -d '{"text": "一个人在跳舞", "duration": 8.0}'
   ```
   
   获得 `task_id`

2. **轮询状态**
   ```bash
   # 每隔 2-5 秒查询一次
   curl http://localhost:8000/status/{task_id}
   ```
   
   直到 `status` 为 `completed` 或 `failed`

3. **下载结果**
   ```bash
   # 方式 1：使用 result_urls（推荐）
   curl -o motion.bvh "{result_urls.bvh}"
   curl -o motion.fbx "{result_urls.fbx}"
   curl -o motion.mp4 "{result_urls.rendered_video}"
   
   # 方式 2：使用 download_url（兼容旧版）
   curl -o motion.fbx "{download_url}"
   ```

### 轮询示例（Shell）

```bash
#!/bin/bash

# 提交任务
RESPONSE=$(curl -s -X POST http://localhost:8000/upload \
  -H "Content-Type: application/json" \
  -d '{"text": "一个人在走路", "duration": 8.0}')

TASK_ID=$(echo $RESPONSE | jq -r '.task_id')
echo "Task ID: $TASK_ID"

# 轮询状态
while true; do
  STATUS=$(curl -s http://localhost:8000/status/$TASK_ID)
  STATE=$(echo $STATUS | jq -r '.status')
  PROGRESS=$(echo $STATUS | jq -r '.progress')
  MESSAGE=$(echo $STATUS | jq -r '.message')
  
  echo "[$STATE] $PROGRESS - $MESSAGE"
  
  if [ "$STATE" == "completed" ] || [ "$STATE" == "failed" ]; then
    break
  fi
  
  sleep 3
done

# 下载结果
if [ "$STATE" == "completed" ]; then
  FBX_URL=$(echo $STATUS | jq -r '.result_urls.fbx')
  curl -o "motion_${TASK_ID}.fbx" "$FBX_URL"
  echo "Downloaded: motion_${TASK_ID}.fbx"
fi
```

### Python 客户端示例

```python
import requests
import time
import json

class MotionClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def generate_motion(self, text: str, duration: float = 8.0):
        """提交动作生成任务"""
        response = requests.post(
            f"{self.base_url}/upload",
            json={"text": text, "duration": duration}
        )
        response.raise_for_status()
        return response.json()["task_id"]
    
    def get_status(self, task_id: str):
        """查询任务状态"""
        response = requests.get(f"{self.base_url}/status/{task_id}")
        response.raise_for_status()
        return response.json()
    
    def wait_for_completion(self, task_id: str, poll_interval: float = 3.0):
        """等待任务完成"""
        while True:
            status = self.get_status(task_id)
            print(f"[{status['status']}] {status.get('progress', 0):.1%} - {status.get('message', '')}")
            
            if status["status"] in ["completed", "failed"]:
                return status
            
            time.sleep(poll_interval)
    
    def download_result(self, task_id: str, output_path: str, file_type: str = "fbx"):
        """下载结果文件"""
        status = self.get_status(task_id)
        
        if status["status"] != "completed":
            raise ValueError(f"Task not completed: {status['status']}")
        
        url = status["result_urls"].get(file_type)
        if not url:
            raise ValueError(f"No {file_type} file available")
        
        response = requests.get(url)
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            f.write(response.content)

# 使用示例
client = MotionClient()

# 生成动作
task_id = client.generate_motion("一个人在跑步", duration=8.0)
print(f"Task ID: {task_id}")

# 等待完成
result = client.wait_for_completion(task_id)

# 下载结果
if result["status"] == "completed":
    client.download_result(task_id, "motion.fbx", "fbx")
    client.download_result(task_id, "motion.mp4", "rendered_video")
    print("Downloaded successfully!")
```

---

## 错误处理

### 常见错误

#### 1. 任务不存在
```json
{
  "detail": "Task not found"
}
```
**状态码**: `404 Not Found`

**原因**: 提供的 `task_id` 不存在或已被清理

**解决**: 检查 `task_id` 是否正确，任务在完成后会保留 24 小时

#### 2. 参数错误
```json
{
  "detail": "Text description cannot be empty"
}
```
**状态码**: `400 Bad Request`

**原因**: 请求参数不符合要求

**解决**: 检查请求参数是否符合 API 规范

#### 3. 任务失败
```json
{
  "task_id": "abc123",
  "status": "failed",
  "message": "Error: ...",
  "error_message": "详细错误信息"
}
```
**状态码**: `200 OK`（状态正常返回，但任务失败）

**原因**: 
- 模型推理失败
- 重定向服务不可用
- 渲染服务不可用
- 存储服务不可用

**解决**: 检查服务日志，确认依赖服务运行正常

#### 4. 任务超时
```json
{
  "task_id": "abc123",
  "status": "failed",
  "message": "Task processing timeout",
  "error_message": "Progress not updated for a long time, marked as timeout"
}
```
**原因**: 任务处理时间超过 60 分钟（默认）

**解决**: 检查服务负载，或增加 `MOTION_TASK_STALE_MIN` 环境变量

---

## 限制说明

### 速率限制
当前版本无速率限制

### 任务限制
- 最大队列长度：无限制
- 单个任务最大处理时间：60 分钟（可配置）
- 任务保留时间：完成后 24 小时
- 最大任务缓存数：1000 个

### 文件大小限制
- BVH 文件：通常 < 1 MB
- FBX 文件：通常 < 5 MB
- 渲染视频：通常 < 50 MB

### 动作生成限制
- 最小时长：2 秒（实际生成 7 秒）
- 最大时长：12 秒（实际生成 10 秒）
- 帧率：30 FPS（固定）

---

## 配置说明

### 环境变量

服务通过环境变量进行配置，可在 `.env.dev` 文件中设置：

```bash
# Blender 服务配置
BLENDER_BASE_URL=http://localhost:8001
RETARGET_SERVICE_URL=http://localhost:8001/retarget
RETARGET_STATUS_URL=http://localhost:8001/retarget_status
RENDER_SERVICE_URL=http://localhost:8001/render_animation
RENDER_STATUS_URL=http://localhost:8001/render_status

# 资源路径
ASSET_PATH=./assets
MOMAX_BOT_PATH=./assets/MomaxBot.fbx

# 队列配置
MOTION_QUEUE_DEFAULT_WORKERS=1    # 默认工作线程数
MOTION_NUM_WORKERS=1               # motion_generation 队列工作线程数
MOTION_POLL_INTERVAL_SEC=5         # 监控轮询间隔（秒）
MOTION_TASK_STALE_MIN=60           # 任务超时时间（分钟）

# 存储配置
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=motion-bvh
MINIO_SECURE=false

# OpenAI 配置（可选，用于提示词优化）
OPENAI_BASE_URL=https://aiping.cn/api/v1
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=DeepSeek-R1-0528

# 服务端口
PORT=8000
```

---

## 性能优化建议

### 1. 增加并发处理能力
```bash
# 同时处理 3 个任务
export MOTION_NUM_WORKERS=3
```

### 2. 调整轮询间隔
```bash
# 更频繁地检查外部服务状态（适合快速响应）
export RETARGET_POLL_INTERVAL=1.0

# 更频繁地监控任务状态
export MOTION_POLL_INTERVAL_SEC=2
```

### 3. 使用本地存储
如果不需要分布式部署，可以使用本地 MinIO 实例以减少网络延迟。

### 4. 批量处理
对于多个任务，可以并行提交，利用队列系统自动调度。

---

## 故障排查

### 服务启动失败
1. 检查端口是否被占用：`netstat -tulpn | grep 8000`
2. 检查依赖服务是否运行：MinIO、Blender 服务
3. 查看启动日志

### 任务一直在队列中
1. 检查工作线程数：`curl http://localhost:8000/queue`
2. 查看服务日志，确认 worker 是否正常运行
3. 重启服务以重新初始化 workers

### 重定向/渲染失败
1. 确认 Blender 服务可访问：`curl http://localhost:8001/health`
2. 检查网络连接
3. 查看 Blender 服务日志

### 下载链接过期
MinIO 预签名 URL 有效期为 24 小时，过期后需要重新获取：
```bash
curl http://localhost:8000/status/{task_id}
```

---

## 更新日志

### v1.1.0 (2025-10-29)
- ✨ 新增异步队列处理系统
- ✨ 新增自动重定向到 FBX 功能
- ✨ 新增自动渲染预览视频功能
- ✨ 新增队列状态查询接口
- ✨ 新增任务列表查询接口
- 🐛 修复任务超时未正确标记的问题
- 📝 完善 API 文档

### v1.0.0 (2025-10-01)
- 🎉 初始版本发布
- ✨ 支持文本到动作生成
- ✨ 支持 BVH 格式输出
- ✨ 支持 LLM 提示词优化

---

## 支持与反馈

如有问题或建议，请通过以下方式联系：

- GitHub Issues: [项目地址]
- 邮箱: [联系邮箱]

---

## 许可证

[许可证信息]
