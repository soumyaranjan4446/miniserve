from __future__ import annotations
import threading
import time

class Metrics:
    def __init__(self): self.lock=threading.Lock(); self.queue_depth=0; self.running=0; self.requests=0; self.completed=0; self.aborted=0; self.output_tokens=0; self.total_e2e_ms=0.; self.total_ttft_ms=0.; self.started_at=time.perf_counter()
    def request_started(self):
        with self.lock: self.requests+=1; self.running+=1
    def request_completed(self,tokens,e2e_ms,ttft_ms):
        with self.lock: self.running=max(0,self.running-1); self.completed+=1; self.output_tokens+=tokens; self.total_e2e_ms+=e2e_ms; self.total_ttft_ms+=ttft_ms
    def request_aborted(self):
        with self.lock: self.running=max(0,self.running-1); self.aborted+=1
    def snapshot(self):
        with self.lock:
            elapsed=time.perf_counter()-self.started_at
            return {"queue_depth":self.queue_depth,"running_requests":self.running,"request_count":self.requests,"completed_requests":self.completed,"aborted_requests":self.aborted,"output_tokens":self.output_tokens,"tokens_per_second":self.output_tokens/elapsed if elapsed else 0.,"mean_e2e_ms":self.total_e2e_ms/self.completed if self.completed else 0.,"mean_ttft_ms":self.total_ttft_ms/self.completed if self.completed else 0.}
