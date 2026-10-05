from __future__ import annotations
import json, os, threading, uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from engine import MiniServeRuntime
from sampling import SamplingParams
from .metrics import Metrics
from .schemas import CompletionRequest, CompletionResponse

MODEL_NAME=os.getenv("MINISERVE_MODEL","Qwen/Qwen2.5-0.5B")
DEVICE=os.getenv("MINISERVE_DEVICE","cuda" if __import__("torch").cuda.is_available() else "cpu")
MAX_QUEUE=int(os.getenv("MINISERVE_MAX_QUEUE","32"))
app=FastAPI(title="MiniServe",version="0.1.0"); metrics=Metrics(); _runtime=None; _lock=threading.Lock()

def load_runtime():
    import torch
    from transformers import AutoTokenizer
    from model import MiniQwen, load_qwen_config
    from weights import load_hf_weights
    tok=AutoTokenizer.from_pretrained(MODEL_NAME); cfg=load_qwen_config(MODEL_NAME); model=MiniQwen(cfg)
    load_hf_weights(model,MODEL_NAME)
    dtype=torch.float16 if DEVICE.startswith("cuda") else torch.float32
    model=model.to(DEVICE,dtype=dtype).eval()
    return MiniServeRuntime(model,tok,DEVICE)

def get_runtime():
    global _runtime
    with _lock:
        if _runtime is None: _runtime=load_runtime()
        return _runtime

def params(p): return SamplingParams(temperature=p.temperature,top_k=p.top_k,top_p=p.top_p,repetition_penalty=p.repetition_penalty,seed=p.seed,stop_token_id=p.stop_token_id,max_new_tokens=p.max_tokens)

@app.get("/health")
def health(): return {"status":"ok","model":MODEL_NAME,"device":DEVICE}
@app.get("/metrics")
def get_metrics(): return metrics.snapshot()

@app.post("/v1/completions",response_model=None)
def completions(payload:CompletionRequest, request:Request):
    if payload.model!="miniserve": raise HTTPException(400,"Only model='miniserve' is supported")
    if metrics.snapshot()["running_requests"] >= MAX_QUEUE: raise HTTPException(429,"MiniServe queue is full")
    runtime=get_runtime(); metrics.request_started(); p=params(payload)
    if payload.stream:
        def events():
            try:
                emitted = 0
                for item in runtime.stream(payload.prompt, p):
                    emitted += 1
                    yield f"data: {json.dumps(item)}\n\n"
                metrics.request_completed(emitted, 0.0, 0.0)
                yield "data: [DONE]\n\n"
            except GeneratorExit:
                metrics.request_aborted()
                raise
            except Exception:
                metrics.request_aborted()
                raise
        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})
    try:
        result=runtime.generate(payload.prompt,p); metrics.request_completed(result.completion_tokens,result.e2e_ms,result.ttft_ms)
        return CompletionResponse(id="cmpl-"+uuid.uuid4().hex,model="miniserve",text=result.text,token_ids=result.token_ids,prompt_tokens=result.prompt_tokens,completion_tokens=result.completion_tokens,ttft_ms=result.ttft_ms,e2e_ms=result.e2e_ms)
    except Exception:
        metrics.request_aborted(); raise

@app.on_event("shutdown")
def shutdown():
    global _runtime
    _runtime=None
