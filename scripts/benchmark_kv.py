import time, torch
from model import MiniQwen,load_qwen_config
from weights import load_hf_weights
from kv_cache import StaticKVCache

def main():
    from transformers import AutoTokenizer
    name="Qwen/Qwen2.5-0.5B"; device="cuda" if torch.cuda.is_available() else "cpu"; cfg=load_qwen_config(name); model=MiniQwen(cfg).to(device).eval(); load_hf_weights(model,name); tok=AutoTokenizer.from_pretrained(name); ids=tok("Explain KV caching.",return_tensors="pt")["input_ids"].to(device)
    for n in [16,32,64,128]:
        x=ids.repeat(1,(n+ids.shape[1]-1)//ids.shape[1])[:,:n]; cache=StaticKVCache(1,cfg.num_hidden_layers,cfg.num_key_value_heads,n+1,cfg.head_dim,device,next(model.parameters()).dtype); model(x,kv_cache=cache); start=time.perf_counter(); model(torch.tensor([[0]],device=device),kv_cache=cache); elapsed=time.perf_counter()-start; print({"seq_len":n,"decode_ms":elapsed*1000})
if __name__=="__main__": main()
