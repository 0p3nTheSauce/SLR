"""
api.py — FastAPI HTTP layer for QueShell
Run with: uvicorn api:app --reload --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, List, Optional, cast

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse # Add this
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).parent))  # ← add this before any local imports

from .core import (
    Que,
    QUE_LOCATIONS,
    SYNONYMS,
    QueLocation,
    TO_RUN, CUR_RUN,
    QueEmpty, QueIdxOOR, QueDupExp, QueBusy, QueException,
    connect_manager,
)

# ---------------------------------------------------------------------------
# Safe criterion parsing (mirrors shell.py)
# ---------------------------------------------------------------------------

SAFE_GLOBALS: dict[str, Any] = {
    "__builtins__": {},
    "len": len, "abs": abs, "round": round,
    "any": any, "all": all, "isinstance": isinstance,
    "str": str, "int": int, "float": float, "list": list,
}


def parse_criterion(expr: str):
    result = eval(expr, SAFE_GLOBALS)  # noqa: S307
    if not callable(result):
        raise ValueError(f"Criterion must be callable, got: {type(result)}")
    return result


# ---------------------------------------------------------------------------
# App lifespan — connect to manager once at startup
# ---------------------------------------------------------------------------

_que = None
_daemon = None
_worker = None
_server_context = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _que, _daemon, _worker, _server_context
    manager = connect_manager()
    _que = manager.get_que()
    _daemon = manager.get_daemon()
    _worker = manager.get_worker()
    _server_context = manager.get_server_context()
    yield


app = FastAPI(title="QueShell API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_loc(loc: str) -> QueLocation:
    resolved = SYNONYMS.get(loc.lower(), loc.lower())
    if resolved not in QUE_LOCATIONS:
        raise HTTPException(status_code=422, detail=f"Unknown location: {loc!r}")
    return resolved  # type: ignore[return-value]


def _http(e: Exception) -> HTTPException:
    if isinstance(e, (QueEmpty, QueIdxOOR)):
        return HTTPException(404, detail=str(e))
    if isinstance(e, (QueDupExp, QueBusy)):
        return HTTPException(409, detail=str(e))
    if isinstance(e, QueException):
        return HTTPException(400, detail=str(e))
    return HTTPException(500, detail=str(e))


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class MoveRequest(BaseModel):
    o_loc: str
    n_loc: str
    oi_idx: int
    of_idx: Optional[int] = None


class ShuffleRequest(BaseModel):
    loc: str
    o_idx: int
    n_idx: int


class RecoverRequest(BaseModel):
    from_loc: str = CUR_RUN
    to_loc: str = TO_RUN
    index: int = 0
    clean_slate: bool = False


class EditRequest(BaseModel):
    loc: str
    idx: int
    keys: List[str]
    value: str
    do_eval: bool = False


class CopyRequest(BaseModel):
    o_loc: str
    o_indexes: List[int] = [0]
    n_loc: str
    n_idx: int = 0
    clean_slate: bool = False


class FindRequest(BaseModel):
    loc: str
    keys: List[str]
    criterion: str  # e.g. "lambda x: x < 5"


class SaveLoadRequest(BaseModel):
    target: str  # "que" | "server"
    timestamp: bool = False
    path: Optional[str] = None


class DaemonStopRequest(BaseModel):
    supervisor: bool = False
    worker: bool = False
    timeout: int = 10
    hard: bool = False


# ---------------------------------------------------------------------------
# Queue — read
# ---------------------------------------------------------------------------

@app.get("/api/queue/{loc}")
def list_runs(
    loc: str,
    sort_keys: List[str] = Query(default=[]),
    reverse: bool = False,
):
    """Return summarised runs for a location."""
    location = resolve_loc(loc)
    # _que = cast(Que, _que)
    try:
        runs = _que.list_runs(location, sort_keys, reverse)
        return [s.model_dump() for s in _que.summarise(runs)]
    except Exception as e:
        raise _http(e)


@app.get("/api/queue/{loc}/{idx}")
def peek_run(loc: str, idx: int):
    """Return the full model_dump for a single run."""
    location = resolve_loc(loc)
    try:
        return _que.peak_run(location, idx).model_dump()
    except Exception as e:
        raise _http(e)


# ---------------------------------------------------------------------------
# Queue — mutations
# ---------------------------------------------------------------------------

@app.delete("/api/queue/{loc}/{idx}")
def remove_run(loc: str, idx: int):
    location = resolve_loc(loc)
    try:
        _que.remove_run(location, idx)
        return {"ok": True}
    except Exception as e:
        raise _http(e)


@app.post("/api/queue/{loc}/clear")
def clear_runs(loc: str):
    location = resolve_loc(loc)
    try:
        _que.clear_runs(location)
        return {"ok": True}
    except Exception as e:
        raise _http(e)


@app.post("/api/queue/move")
def move_run(req: MoveRequest):
    try:
        _que.move(resolve_loc(req.o_loc), resolve_loc(req.n_loc), req.oi_idx, req.of_idx)
        return {"ok": True}
    except Exception as e:
        raise _http(e)


@app.post("/api/queue/shuffle")
def shuffle_run(req: ShuffleRequest):
    try:
        _que.shuffle(resolve_loc(req.loc), req.o_idx, req.n_idx)
        return {"ok": True}
    except Exception as e:
        raise _http(e)


@app.post("/api/queue/recover")
def recover_run(req: RecoverRequest):
    try:
        _que.recover_run(
            from_loc=resolve_loc(req.from_loc),
            to_loc=resolve_loc(req.to_loc),
            index=req.index,
            clean_slate=req.clean_slate,
        )
        return {"ok": True}
    except Exception as e:
        raise _http(e)


@app.post("/api/queue/edit")
def edit_run(req: EditRequest):
    try:
        _que.edit_run(resolve_loc(req.loc), req.idx, req.keys, req.value, req.do_eval)
        return {"ok": True}
    except Exception as e:
        raise _http(e)


@app.post("/api/queue/copy")
def copy_runs(req: CopyRequest):
    try:
        _que.copy_runs(
            resolve_loc(req.o_loc), req.o_indexes,
            resolve_loc(req.n_loc), req.n_idx, req.clean_slate,
        )
        return {"ok": True}
    except Exception as e:
        raise _http(e)


@app.post("/api/queue/find")
def find_runs(req: FindRequest):
    """
    Filter runs by a criterion lambda.
    Callable can't cross the proxy boundary, so filtering runs locally
    after fetching from the manager.
    """
    location = resolve_loc(req.loc)
    try:
        criterion = parse_criterion(req.criterion)
    except (ValueError, SyntaxError) as e:
        raise HTTPException(422, detail=f"Invalid criterion: {e}")
    try:
        runs = _que.list_runs(location)
        idxs, matched = [], []
        for i, run in enumerate(runs):
            val = Que.get_nested(run, req.keys)
            if criterion(val):
                idxs.append(i)
                matched.append(run)
        summaries = Que.summarise(matched)  # type: ignore[arg-type]
        return {"indexes": idxs, "runs": [s.model_dump() for s in summaries]}
    except Exception as e:
        raise _http(e)


# ---------------------------------------------------------------------------
# Server status
# ---------------------------------------------------------------------------

@app.get("/api/status")
def get_status():
    try:
        return _server_context.get_state().model_dump()
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------

@app.post("/api/daemon/start")
def daemon_start():
    try:
        _daemon.start_supervisor()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.post("/api/daemon/stop")
def daemon_stop(req: DaemonStopRequest):
    try:
        if req.supervisor:
            _daemon.stop_supervisor(timeout=req.timeout, hard=req.hard, stop_worker=req.worker)
        else:
            _daemon.stop_worker(timeout=req.timeout, hard=req.hard)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

@app.post("/api/worker/cleanup")
def worker_cleanup():
    try:
        _worker.cleanup()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

@app.post("/api/save")
def save_state(req: SaveLoadRequest):
    try:
        if req.target == "que":
            _que.save_state(out_path=req.path, timestamp=req.timestamp)
        elif req.target == "server":
            _server_context.save_state()
        else:
            raise HTTPException(422, detail=f"Unknown target: {req.target!r}")
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.post("/api/load")
def load_state(req: SaveLoadRequest):
    try:
        if req.target == "que":
            _que.load_state(req.path)
        elif req.target == "server":
            _server_context.load_state()
        else:
            raise HTTPException(422, detail=f"Unknown target: {req.target!r}")
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=str(e))
    
@app.get("/", include_in_schema=False)
async def serve_gui():
    # This assumes gui.html is in the same folder as api.py
    return FileResponse("que/gui.html")