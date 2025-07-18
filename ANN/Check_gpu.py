# Save as test_gpu.py
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ Using GPU:", torch.cuda.get_device_name(0))
    a = torch.rand(5000, 5000, device=device)
    b = torch.mm(a, a)
    print("✅ Matrix multiplication completed on GPU.")
else:
    print("❌ CUDA GPU not available.")
