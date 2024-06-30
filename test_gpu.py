import torch
import torch
import torch.nn as nn
import torch.optim as optim

print("PyTorch version:", torch.version)
print("Is CUDA available?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA (ROCm) is available.")
    print("Number of CUDA devices:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"Device {i} name:", torch.cuda.get_device_name(i))
else:
    print("CUDA (ROCm) is not available.")

class SimpleModel(nn.Module):
    def init(self):
        super(SimpleModel, self).init()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

models = []
optimizers = []
devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]

for i, device in enumerate(devices):
    model = SimpleModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    models.append(model)
    optimizers.append(optimizer)

inputs = [torch.randn(64, 10).to(device) for device in devices]
outputs = []

for i, model in enumerate(models):
    output = model(inputs[i])
    outputs.append(output)
    print(f'Output from GPU {i}: {output}')
