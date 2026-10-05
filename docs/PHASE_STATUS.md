# MiniServe Release Audit

## Verified locally

- Python syntax compilation: passed for the full repository.
- Pytest: 23 tests passed.
- Sampling smoke test: passed.
- Phase 5 scheduler/paged-cache smoke test: passed.
- Speculative decoding smoke test: passed.
- Static batching benchmark: passed.
- Paged block-size benchmark: passed.
- Phase 8 server module import and Pydantic schema: passed.

## External-model validation

The repository includes Qwen2.5-0.5B Hugging Face parity and baseline scripts. Those scripts require the `transformers` package and access to the model files. The build environment used to package this release did not have `transformers` cached and had no network access, so the real Qwen weight/logit parity run could not be executed here.

After installing requirements on the target machine, run:

```powershell
python -m scripts.inspect_model
python -m scripts.test_qwen
python -m scripts.run_baseline
pytest -v
```

The code fails loudly if the Hugging Face state dictionary has missing keys, unexpected keys, or shape mismatches.

## Intentional scope boundaries

- Phase 5 contains a correctness-first paged KV memory subsystem and continuous request scheduler. It is not a custom CUDA/Triton PagedAttention kernel.
- Phase 6 INT8 is a reference QLinear implementation; it may be slower than FP16/FP32 because the current path converts INT8 weights to floating-point for the matrix multiplication. This is intentional and provides a measured baseline for Phase 9.
- Phase 7 is optional and uses transparent greedy speculative decoding without claiming a production optimized verifier.
- Phase 8 serves through the MiniServe model/runtime path. It does not use `model.generate()` for generation.
- Phase 9 profiles the MiniServe path and records before/after benchmark data; it does not fabricate a speedup.
