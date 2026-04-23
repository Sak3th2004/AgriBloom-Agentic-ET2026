"""Export trained EfficientNet-B4 to ONNX format for offline inference."""
import torch
from efficientnet_pytorch import EfficientNet
import json
import os

model_dir = "models/checkpoints/efficientnet_b4_indian"
label_path = os.path.join(model_dir, "class_labels.json")

# Load class labels
with open(label_path) as f:
    labels = json.load(f)
num_classes = len(labels)
print(f"Classes: {num_classes}")

# Load checkpoint
checkpoint = torch.load(os.path.join(model_dir, "best_model.pth"), map_location="cpu")
print(f"Checkpoint keys: {list(checkpoint.keys())}")
val_acc = checkpoint.get("val_acc", "N/A")
print(f"Val Accuracy: {val_acc}")

# Build model and load weights
model = EfficientNet.from_pretrained("efficientnet-b4", num_classes=num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("Model loaded successfully!")

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
onnx_path = os.path.join(model_dir, "agribloom_efficientnet_b4.onnx")
torch.onnx.export(
    model, dummy_input, onnx_path,
    input_names=["image"],
    output_names=["prediction"],
    dynamic_axes={"image": {0: "batch"}, "prediction": {0: "batch"}},
    opset_version=13,
)

size_mb = os.path.getsize(onnx_path) / 1e6
print(f"ONNX exported: {onnx_path} ({size_mb:.1f} MB)")
print("Done!")
