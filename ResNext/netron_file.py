import os
import torch
import yaml
import torch.nn as nn
from PulsoVital.convolutional.ResNext.src.resnext import ConvResNet

# Load config file
rootdir = os.path.dirname(os.path.realpath(__file__))
config_path = r'C:\Users\maria\Desktop\GITHUB\PulsoVital-main\PulsoVital\convolutional\ConvResNet\config.yaml'

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Model hyperparameters
model_config = config["model"]
in_channels = 1
out_features = 4
sequence_length = config["data"]["length"]

# Device and dtype
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32  # safer than bfloat16 for export

# Build model
model = ConvResNet(
    in_channels=in_channels,
    out_features=out_features,
    activation=nn.ReLU,
    **model_config
).to(device=device, dtype=dtype)

model.eval()

# Dummy input with correct shape: (batch_size=1, channels=1, sequence_length=1000)
dummy_input = torch.randn(1, in_channels, sequence_length, device=device, dtype=dtype)

# Export to ONNX
output_path = os.path.join(rootdir, "convresnet.onnx")

torch.onnx.export(
    model,
    dummy_input,
    output_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=17,
    export_params=True
)

print(f"ONNX model exported successfully to:\n{output_path}")
print("You can now open it with https://netron.app or via Netron CLI.")

# Optional: launch Netron if installed locally
try:
    import netron
    netron.start(output_path)
except ImportError:
    print("Netron is not installed. To open locally, run: pip install netron")
    print("Then run: netron convresnet.onnx")


# Ruta donde deseas guardar el modelo ONNX
onnx_path = r'C:\Users\maria\Desktop\GITHUB\PulsoVital-main\PulsoVital\convolutional\ConvResNet\runs\20250416-0852\convresnet.onnx'
