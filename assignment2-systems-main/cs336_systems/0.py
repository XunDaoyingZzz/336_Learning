import torch
import time

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"cuDNN版本: {torch.backends.cudnn.version()}")

s=torch.tensor(0,dtype=torch.float32)
T1=time.time()
for i in range(1000):
    s+=torch.tensor(0.01,dtype=torch.float32)
print(s)
T2=time.time()
print(f"用时{T2-T1}")


s=torch.tensor(0,dtype=torch.float16)
T1=time.time()
for i in range(1000):
    s+=torch.tensor(0.01,dtype=torch.float16)
print(s)
T2=time.time()
print(f"用时{T2-T1}")


s=torch.tensor(0,dtype=torch.float32)
T1=time.time()
for i in range(1000):
    s+=torch.tensor(0.01,dtype=torch.float16)
print(s)
T2=time.time()
print(f"用时{T2-T1}")

s=torch.tensor(0,dtype=torch.float32)
T1=time.time()
for i in range(1000):
    s+=torch.tensor(0.01,dtype=torch.float16)
print(s)
T2=time.time()
print(f"用时{T2-T1}")