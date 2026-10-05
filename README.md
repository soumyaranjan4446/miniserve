# MiniServe

MiniServe is a from-scratch, correctness-first LLM inference and serving engine built around Qwen2.5-0.5B weights. It implements the model path, KV caching, sampling, batching, paged KV memory, quantization, speculative decoding, HTTP serving, and profiling without replacing the core inference path with Hugging Face generation.

## Phases

- Phase 0: environment and measurement foundation
- Phase 1: Qwen transformer from scratch + HF weight parity
- Phase 2: static KV cache
- Phase 3: sampling
- Phase 4: static batching and padding analysis
- Phase 5: paged KV primitives + continuous request scheduler
- Phase 6: per-output-channel symmetric INT8 QLinear
- Phase 7: optional exact-greedy speculative decoding
- Phase 8: OpenAI-style `/v1/completions` + SSE
- Phase 9: profiler + benchmark harness

Phase 7 is optional in the project specification. Phase 9 deliberately reports profiler evidence before recommending an optimization; it does not claim a speedup that was not measured.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the complete local suite:

```powershell
pytest -v
```

HF parity/benchmark scripts need internet access or a locally cached model:

```powershell
python -m scripts.inspect_model
python -m scripts.run_baseline
python -m scripts.test_qwen
```

## Server

The server lazily loads the model only when a completion request arrives.

```powershell
python -m scripts.run_server
```

Then POST JSON to `http://127.0.0.1:8000/v1/completions`.

## Design rule

The implementation is intentionally simple. Correctness comes before kernels, CUDA graphs, or custom Triton code. Phase 9 is the place to make one measured optimization after profiling.
