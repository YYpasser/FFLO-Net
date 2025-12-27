import onnx
from onnxsim import simplify
# Load the ONNX model
model = onnx.load("rt_FFLONet.onnx")
# Simplify the model
model_simplified, check = simplify(model)
# Ensure the simplified model is valid
assert check, "Simplified ONNX model could not be validated"
# Save or use the simplified model
onnx.save(model_simplified, "rt_FFLONet_simplified.onnx")