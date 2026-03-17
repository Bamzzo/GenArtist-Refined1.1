"""
GenArtist FastAPI 后端入口。
将重型图像生成任务放入线程池执行，避免阻塞异步事件循环。
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import WORK_DIR
from demo_t2i import run_agent_pipeline

# --- 应用实例与中间件 ---
app = FastAPI(title="GenArtist API", description="Text-to-Image agent pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 启动前确保输出目录存在
WORK_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(WORK_DIR)), name="outputs")


@app.exception_handler(422)
async def unprocessable_handler(request: Request, exc):
    """422 时返回更明确的提示：多为请求体 JSON 格式错误或缺少 prompt。"""
    try:
        body = await request.body()
        if body and body[:1] in (b"{", b"["):
            return JSONResponse(
                status_code=422,
                content={
                    "detail": "请求体必须是合法 JSON，且包含 prompt 字段。示例: {\"prompt\": \"描述文字\"}",
                    "raw_error": getattr(exc, "detail", str(exc)),
                },
            )
    except Exception:
        pass
    return JSONResponse(status_code=422, content={"detail": getattr(exc, "detail", str(exc))})


# --- 请求模型 ---
class GenerateRequest(BaseModel):
    prompt: str


def _image_path_to_url(abs_or_rel_path: str) -> str:
    """将本地路径（绝对或相对）转为可通过 /outputs 访问的 URL 路径。"""
    p = Path(abs_or_rel_path).resolve()
    try:
        rel = p.relative_to(WORK_DIR.resolve())
    except ValueError:
        # 不在 WORK_DIR 下，只取文件名，避免暴露任意路径
        rel = p.name
    return "/outputs/" + str(rel).replace("\\", "/")


@app.post("/api/v1/generate_direct")
async def api_generate_direct(request: GenerateRequest):
    """
    直连生图：仅跑 SDXL text_to_image，不经过 LLM 规划。用于自愈测试与快速出图。
    """
    if not (request.prompt or "").strip():
        raise HTTPException(status_code=400, detail="prompt must be non-empty")
    import os.path as osp
    out_name = "direct.png"
    out_path = str(WORK_DIR / out_name)
    try:
        from agent_tool_generate import run_generate_tool
        cmd = {
            "tool": "text_to_image_SDXL",
            "input": {"text": request.prompt.strip()},
            "output": out_path,
        }
        result_path = await run_in_threadpool(run_generate_tool, cmd)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"generate_direct failed: {e!s}")
    if not result_path or not osp.isfile(result_path):
        raise HTTPException(status_code=500, detail="no image file produced")
    return {
        "status": "success",
        "image_url": _image_path_to_url(result_path),
        "image_path": result_path,
    }


@app.post("/api/v1/generate")
async def api_generate(request: GenerateRequest):
    """
    提交文本提示，在线程池中执行 run_agent_pipeline，返回生成结果或错误。
    """
    if not (request.prompt or "").strip():
        raise HTTPException(status_code=400, detail="prompt must be non-empty")

    try:
        result = await run_in_threadpool(
            run_agent_pipeline,
            request.prompt.strip(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"pipeline execution failed: {e!s}",
        )

    if result["status"] == "error":
        raise HTTPException(
            status_code=422,
            detail=result.get("error_msg", "unknown error"),
        )

    # 成功：将 image_path 转为前端可访问的 URL
    image_path = result.get("image_path") or ""
    image_url = _image_path_to_url(image_path) if image_path else ""
    return {
        "status": "success",
        "image_url": image_url,
        "image_path": image_path,
        "history": result.get("history", []),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
