from pydantic import BaseModel, Field

class CompletionRequest(BaseModel):
    model:str="miniserve"
    prompt:str
    max_tokens:int=Field(128,ge=0,le=4096)
    temperature:float=Field(0.7,ge=0)
    top_k:int|None=Field(None,gt=0)
    top_p:float|None=Field(0.9,gt=0,le=1)
    repetition_penalty:float=Field(1.0,gt=0)
    seed:int|None=None
    stop_token_id:int|None=None
    stream:bool=False

class CompletionResponse(BaseModel):
    id:str; object:str="text_completion"; model:str; text:str; token_ids:list[int]
    prompt_tokens:int; completion_tokens:int; ttft_ms:float; e2e_ms:float
