import torch,time
from quantization import QLinear,model_storage_bytes

def main():
    layer=torch.nn.Linear(1024,1024); q=QLinear.from_linear(layer); x=torch.randn(32,1024)
    for _ in range(3): layer(x); q(x)
    def bench(fn):
        t=time.perf_counter()
        for _ in range(20): fn()
        return (time.perf_counter()-t)/20*1000
    print({"fp32_ms":bench(lambda:layer(x)),"int8_reference_ms":bench(lambda:q(x)),"fp32_bytes":model_storage_bytes(layer),"int8_bytes":model_storage_bytes(q)})
if __name__=="__main__": main()
