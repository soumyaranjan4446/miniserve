import json
from bench.baseline_hf import run_baseline
if __name__=="__main__": print(json.dumps(run_baseline(),indent=2))
