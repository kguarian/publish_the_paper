import onnxmltools
import coremltools

model_filepath = "obj_det/visual_iter_1400.mlmodel"

# Load a Core ML model
coreml_model = coremltools.utils.load_spec(model_filepath)

# Convert the Core ML model into ONNX
onnx_model = onnxmltools.convert_coreml(coreml_model, 'onnx object detection model')
onnxmltools.utils.visualize_model(onnx_model)
# Save as protobuf
# onnxmltools.utils.save_model(onnx_model, 'onnx_obj_det.onnx')
