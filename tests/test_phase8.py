def test_schema_defaults():
    from server.schemas import CompletionRequest
    p=CompletionRequest(prompt="hello"); assert p.model=="miniserve" and p.max_tokens==128 and p.stream is False

def test_metrics():
    from server.metrics import Metrics
    m=Metrics(); m.request_started(); m.request_completed(3,30,5); s=m.snapshot(); assert s["completed_requests"]==1 and s["output_tokens"]==3
