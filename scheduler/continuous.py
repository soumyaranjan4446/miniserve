from collections import deque
from .request import Request, RequestState

class ContinuousBatchingScheduler:
    def __init__(self, max_running_requests, paged_cache):
        if max_running_requests <= 0: raise ValueError("max_running_requests must be positive")
        self.max_running_requests=max_running_requests; self.paged_cache=paged_cache
        self.waiting=deque(); self.running={}; self.finished={}; self.aborted={}; self.step_count=0

    def add_request(self, request: Request):
        if request.request_id in self.running or request.request_id in self.finished or request.request_id in self.aborted or any(r.request_id==request.request_id for r in self.waiting):
            raise ValueError(f"duplicate request ID: {request.request_id}")
        request.arrival_step=self.step_count; self.waiting.append(request)

    def admit_requests(self):
        while self.waiting and len(self.running) < self.max_running_requests:
            r = self.waiting.popleft()
            self.paged_cache.create_request(r.request_id)
            try:
                self.paged_cache.ensure_capacity(r.request_id, r.prompt_length)
            except RuntimeError:
                self.paged_cache.delete_request(r.request_id)
                self.waiting.appendleft(r)
                break
            r.state = RequestState.RUNNING
            self.running[r.request_id] = r

    def next_batch(self): self.admit_requests(); return list(self.running.values())
    def step(self): self.step_count += 1; return self.next_batch()

    def finish_request(self, request_id):
        r=self.running.pop(request_id); r.mark_finished(self.step_count); self.paged_cache.delete_request(request_id); self.finished[request_id]=r
        self.admit_requests()

    def abort_request(self, request_id, reason):
        r=self.running.pop(request_id); r.abort(reason); self.paged_cache.delete_request(request_id); self.aborted[request_id]=r
        self.admit_requests()

    def waiting_count(self): return len(self.waiting)
    def running_count(self): return len(self.running)
    def finished_count(self): return len(self.finished)
    def aborted_count(self): return len(self.aborted)
    def is_empty(self): return not self.waiting and not self.running
