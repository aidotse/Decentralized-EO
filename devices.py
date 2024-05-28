import torch

print("PyTorch version:", torch.__version__)
print("Is CUDA available?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA (ROCm) is available.")
    print("Number of CUDA devices:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"Device {i} name:", torch.cuda.get_device_name(i))
else:
    print("CUDA (ROCm) is not available.")
