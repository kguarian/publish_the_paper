import coremltools as ct
import onnxmltools
from coremltools import converters

# Load a Core ML model
model_filepath = "obj_det/visual_iter_1400.mlmodel"
coreml_model = ct.models.MLModel(model_filepath)

ct.

# Convert to ONNX
onnx_model_path = "onnx_obj_det.onnx"
# Convert the Core ML model to ONNX
onnx_model = onnxmltools.convert_coreml(coreml_model)

print(f"ONNX model saved at {onnx_model_path}")