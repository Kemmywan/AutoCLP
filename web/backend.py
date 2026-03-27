# web/backend.py
"""
CPL-Clinic Web Backend - FastAPI
提供分步执行的REST API，支持SSE流式输出
"""

import asyncio
import json
import os
import sys
import uuid
from dataclasses import asdict
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from ambient import MultimodalAdapter
from commander import CommanderLLM, BranchNode, flatten_tasks
from commander.task_schema import BaseTask, TaskType, TaskStatus
from cpl import CPLGenerator, CPLInterpreter
from llm_manager import LLMManager

app = FastAPI(title="CPL-Clinic", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ==================== 全局单例 ====================

_llm_mgr: LLMManager | None = None
_commander: CommanderLLM | None = None
_adapter = MultimodalAdapter()
_generator = CPLGenerator()
_interpreter = CPLInterpreter()


def _init_llm():
    global _llm_mgr, _commander
    if _llm_mgr is None:
        _llm_mgr = LLMManager()
        _commander = CommanderLLM(agent=_llm_mgr.commander_agent)


# ==================== 会话管理 ====================

class Session:
    """一次完整的pipeline会话"""
    def __init__(self, session_id: str, dialogue: str):
        self.session_id = session_id
        self.dialogue = dialogue
        self.raw_data = None
        self.label = ""
        self.items = []        # list[BaseTask | BranchNode]
        self.tasks = []        # flattened BaseTask list
        self.cpl_text = ""
        self.cpl_script = None
        self.plan = None
        self.report = None
        self.pending_pairs = []  # 待确认的 (record, diagnostic) 二元组
        self.stage = "init"    # init → classified → decomposed → cpl → executed
        self.cancelled = False


_sessions: dict[str, Session] = {}


# ==================== 请求模型 ====================

class DialogueInput(BaseModel):
    dialogue: str

class CPLEditInput(BaseModel):
    session_id: str
    cpl_text: str

class SessionAction(BaseModel):
    session_id: str

class PairsConfirmInput(BaseModel):
    session_id: str
    pairs: list[dict]   # [{"record": "...", "diagnostic": "..."}]


# ==================== API 端点 ====================

@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/api/start")
async def start_session(req: DialogueInput):
    """创建会话，预处理输入"""
    _init_llm()
    sid = str(uuid.uuid4())[:8]
    session = Session(sid, req.dialogue.strip())
    session.raw_data = _adapter.ingest_from_string(session.dialogue)
    session.stage = "init"
    _sessions[sid] = session
    return {"session_id": sid, "char_count": len(session.dialogue)}


@app.post("/api/classify")
async def classify(req: SessionAction):
    """Step 0: 标签分类"""
    _init_llm()
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "会话不存在")
    if session.cancelled:
        raise HTTPException(400, "会话已终止")

    label = await _commander.classify(session.raw_data)
    session.label = label
    session.stage = "classified"
    return {"label": label}


@app.post("/api/decompose")
async def decompose(req: SessionAction):
    """Step 1: 任务分解（含分支）"""
    _init_llm()
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "会话不存在")
    if session.cancelled:
        raise HTTPException(400, "会话已终止")

    items = await _commander.decompose(session.raw_data)
    session.items = items
    session.tasks = flatten_tasks(items)
    session.stage = "decomposed"

    return {
        "label": session.label,
        "task_count": len(session.tasks),
        "branch_count": sum(1 for it in items if isinstance(it, BranchNode)),
        "tasks": _serialize_items(items),
    }


@app.post("/api/generate_cpl")
async def generate_cpl(req: SessionAction):
    """Step 2: 生成CPL脚本"""
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "会话不存在")
    if session.cancelled:
        raise HTTPException(400, "会话已终止")

    cpl_text = _generator.render(session.items, pathway_name="门诊处理路径")
    session.cpl_text = cpl_text
    session.stage = "cpl"
    return {"cpl": cpl_text}


@app.post("/api/update_cpl")
async def update_cpl(req: CPLEditInput):
    """用户编辑CPL后更新"""
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "会话不存在")
    session.cpl_text = req.cpl_text
    return {"status": "ok"}


@app.post("/api/execute")
async def execute(req: SessionAction):
    """Step 3: 解释CPL + 执行Plan（SSE流式返回每步结果）"""
    _init_llm()
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "会话不存在")
    if session.cancelled:
        raise HTTPException(400, "会话已终止")

    async def event_generator():
        try:
            # 解释CPL
            plan = _interpreter.interpret(session.cpl_text)
            session.plan = plan

            yield {
                "event": "plan",
                "data": json.dumps({
                    "pathway_name": plan.pathway_name,
                    "total_calls": len(plan.calls),
                    "agent_calls": len(plan.agent_calls_only()),
                }, ensure_ascii=False)
            }

            # 逐步执行（支持ConditionalBlock）
            from cpl.interpreter import AgentCall, ConditionalBlock, CallType
            variables = {"dialogue": session.raw_data.content}
            results = []

            async def execute_items_sse(items):
                """递归执行执行项，yield SSE事件"""
                for item in items:
                    if session.cancelled:
                        return
                    if isinstance(item, AgentCall):
                        async for ev in execute_call_sse(item):
                            yield ev
                    elif isinstance(item, ConditionalBlock):
                        async for ev in execute_conditional_sse(item):
                            yield ev

            async def execute_call_sse(call):
                """执行单个AgentCall并yield SSE事件"""
                if session.cancelled:
                    yield {
                        "event": "cancelled",
                        "data": json.dumps({"message": "用户终止执行"}, ensure_ascii=False)
                    }
                    return

                txn_id = str(uuid.uuid4())
                call.transaction_id = txn_id
                call.status = "running"
                call.started_at = datetime.now().isoformat()

                yield {
                    "event": "step_start",
                    "data": json.dumps({
                        "step": call.step_number,
                        "label": call.step_label,
                        "agent": call.agent_name,
                        "call_type": call.call_type.value,
                        "task_type": call.task_type.value if call.task_type else "",
                    }, ensure_ascii=False)
                }

                result_text = ""
                status = "done"
                error = ""

                if call.call_type == CallType.AGENT and call.task_type is not None:
                    try:
                        result_text = await _llm_mgr._pool._execute_agent_call(call, variables)
                        if call.variable_name:
                            variables[call.variable_name] = result_text
                    except Exception as e:
                        status = "failed"
                        error = str(e)
                elif call.call_type == CallType.RAG:
                    record_text = variables.get("medical_record", variables.get("patient_profile", ""))
                    diag_text = variables.get("diagnostic", "")
                    if record_text and diag_text:
                        session.pending_pairs.append({"record": record_text, "diagnostic": diag_text})
                        result_text = "[RAG] 二元组已暂存，等待医生确认后归档"
                    else:
                        result_text = f"[RAG] 跳过归档: record={'有' if record_text else '无'}, diagnostic={'有' if diag_text else '无'}"
                elif call.call_type == CallType.EXAM:
                    result_text = f"[Exam] {call.agent_name} 检查请求已记录"
                else:
                    result_text = f"[{call.call_type.value}] 已处理"

                call.status = status
                call.result = result_text
                call.error = error
                call.finished_at = datetime.now().isoformat()

                results.append({
                    "step": call.step_number,
                    "label": call.step_label,
                    "agent": call.agent_name,
                    "task_type": call.task_type.value if call.task_type else "",
                    "status": status,
                    "result": result_text[:2000],
                    "error": error,
                })

                yield {
                    "event": "step_done",
                    "data": json.dumps(results[-1], ensure_ascii=False)
                }

            async def execute_conditional_sse(block):
                """执行ConditionalBlock，评估条件选择分支"""
                yield {
                    "event": "conditional_start",
                    "data": json.dumps({
                        "step": block.step_number,
                        "label": block.step_label,
                        "branches": len(block.branches),
                    }, ensure_ascii=False)
                }

                matched = False
                for condition_expr, branch_items in block.branches:
                    result = _llm_mgr._pool._evaluate_condition(condition_expr, variables)
                    if result:
                        matched = True
                        yield {
                            "event": "branch_selected",
                            "data": json.dumps({
                                "step": block.step_number,
                                "condition": condition_expr,
                                "matched": True,
                            }, ensure_ascii=False)
                        }
                        async for ev in execute_items_sse(branch_items):
                            yield ev
                        break

                if not matched and block.else_items:
                    yield {
                        "event": "branch_selected",
                        "data": json.dumps({
                            "step": block.step_number,
                            "condition": "ELSE",
                            "matched": True,
                        }, ensure_ascii=False)
                    }
                    async for ev in execute_items_sse(block.else_items):
                        yield ev

            # 执行所有项
            async for ev in execute_items_sse(plan.calls):
                yield ev
                if session.cancelled:
                    yield {
                        "event": "cancelled",
                        "data": json.dumps({"message": "用户终止执行"}, ensure_ascii=False)
                    }
                    return

            # 最终变量空间就是归档结果
            session.stage = "executed"
            archive_data = {}
            for k, v in variables.items():
                if k != "dialogue":
                    archive_data[k] = v[:1000] if isinstance(v, str) else str(v)[:1000]

            yield {
                "event": "complete",
                "data": json.dumps({
                    "total": len(results),
                    "succeeded": sum(1 for r in results if r["status"] == "done"),
                    "failed": sum(1 for r in results if r["status"] == "failed"),
                    "archive": archive_data,
                    "pending_pairs": session.pending_pairs,
                }, ensure_ascii=False)
            }

        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}, ensure_ascii=False)
            }

    return EventSourceResponse(event_generator())


@app.post("/api/confirm_pairs")
async def confirm_pairs(req: PairsConfirmInput):
    """医生确认编辑后的症状-诊断二元组，存入RAG"""
    _init_llm()
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "会话不存在")

    saved = 0
    for pair in req.pairs:
        record = pair.get("record", "").strip()
        diagnostic = pair.get("diagnostic", "").strip()
        if record and diagnostic:
            _llm_mgr._pool.rag.add_pair(record=record, diagnostic=diagnostic)
            saved += 1
    if saved:
        _llm_mgr._pool.rag.save_index()
    session.pending_pairs = []  # 清空待确认
    return {"saved": saved, "total": len(_llm_mgr._pool.rag)}


@app.post("/api/cancel")
async def cancel(req: SessionAction):
    """终止当前会话"""
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "会话不存在")
    session.cancelled = True
    return {"status": "cancelled"}


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """获取会话状态"""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "会话不存在")
    return {
        "session_id": session.session_id,
        "stage": session.stage,
        "label": session.label,
        "cancelled": session.cancelled,
    }


# ==================== 序列化工具 ====================

def _serialize_items(items: list) -> list[dict]:
    """将list[BaseTask | BranchNode]序列化为JSON可输出格式"""
    result = []
    for item in items:
        if isinstance(item, BranchNode):
            result.append({
                "type": "branch",
                "condition": item.condition,
                "branches": [
                    {
                        "condition_value": cv,
                        "tasks": [_serialize_task(t) for t in tasks]
                    }
                    for cv, tasks in item.branches
                ],
                "else_tasks": [_serialize_task(t) for t in item.else_tasks],
            })
        elif isinstance(item, BaseTask):
            result.append(_serialize_task(item))
    return result


def _serialize_task(task: BaseTask) -> dict:
    """将BaseTask序列化"""
    d = {}
    for k, v in task.__dict__.items():
        if k in ("override_log", "transaction_id"):
            continue
        if isinstance(v, TaskType):
            d[k] = v.value
        elif isinstance(v, TaskStatus):
            d[k] = v.value
        else:
            d[k] = v
    return d


# ==================== 启动 ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
