"""FastAPI 入口：/api/solve、/api/shutdown-analysis 负责校验与求解，
其余路径提供原生前端静态页面。"""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .solver import (
    SolverNotConverged,
    ValidationError,
    analyze_pipe_shutdowns,
    solve_network,
    validate_payload,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="闭环供水管网平衡台", version="1.0.0")


def _reject_constant(value: str) -> Any:
    """拒绝 JSON 标准之外的 NaN / Infinity / -Infinity 记号。"""
    raise ValueError(f"非法的 JSON 数值记号：{value}")


async def _parse_payload(request: Request) -> tuple[Any, JSONResponse | None]:
    """读取并解析请求体；失败时返回 (None, 400 响应)。"""
    raw = await request.body()
    if not raw:
        return None, JSONResponse(status_code=400, content={
            "ok": False, "error": "请求体为空，请先填写节点与管段数据。"})
    try:
        payload = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    except (ValueError, UnicodeDecodeError) as exc:
        return None, JSONResponse(status_code=400, content={
            "ok": False,
            "error": f"请求体不是合法 JSON：{exc}。请确认所有输入均为有限数值。"})
    return payload, None


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/solve")
async def solve(request: Request) -> JSONResponse:
    payload, bad = await _parse_payload(request)
    if bad is not None:
        return bad

    try:
        network = validate_payload(payload)
        result = solve_network(network)
    except ValidationError as exc:
        return JSONResponse(status_code=422,
                            content={"ok": False, "error": str(exc)})
    except SolverNotConverged as exc:
        return JSONResponse(status_code=422,
                            content={"ok": False, "error": str(exc)})

    return JSONResponse(content={"ok": True, "result": result})


@app.post("/api/shutdown-analysis")
async def shutdown_analysis(request: Request) -> JSONResponse:
    """逐根停管后果评估：校验与基线求解失败返回 422/400；
    单个停管情景不收敛不影响整体响应，只在该情景中标明。"""
    payload, bad = await _parse_payload(request)
    if bad is not None:
        return bad

    try:
        network = validate_payload(payload)
        result = analyze_pipe_shutdowns(network)
    except ValidationError as exc:
        return JSONResponse(status_code=422,
                            content={"ok": False, "error": str(exc)})
    except SolverNotConverged as exc:
        return JSONResponse(status_code=422,
                            content={"ok": False, "error": str(exc)})

    return JSONResponse(content={"ok": True, "result": result})


# 静态资源（index.html / style.css / app.js）——该路由必须最后注册
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
