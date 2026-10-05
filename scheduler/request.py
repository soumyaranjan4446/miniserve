from dataclasses import dataclass, field
from enum import Enum

class RequestState(Enum):
    WAITING="waiting"; RUNNING="running"; FINISHED="finished"; ABORTED="aborted"

@dataclass
class Request:
    request_id: str
    input_ids: object
    max_new_tokens: int
    state: RequestState = RequestState.WAITING
    generated_tokens: list[int] = field(default_factory=list)
    arrival_step: int = 0
    finished_step: int | None = None
    error: str | None = None
    @property
    def prompt_length(self): return int(self.input_ids.shape[-1])
    @property
    def output_length(self): return len(self.generated_tokens)
    @property
    def total_length(self): return self.prompt_length + self.output_length
    def add_token(self, token_id): self.generated_tokens.append(int(token_id))
    def mark_finished(self, step): self.state= RequestState.FINISHED; self.finished_step=step
    def abort(self, reason): self.state=RequestState.ABORTED; self.error=reason
