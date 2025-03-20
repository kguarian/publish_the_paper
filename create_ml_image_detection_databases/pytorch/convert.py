import coremltools as ct

filepath = 'Vbd3 2 copy Iteration 50.mlpackage'

model = ct.models.MLModel(filepath)

print("MODEL\n\n", model)

s=model.get_spec()
s=ct.models.MLModel(s)
if s is not None:
    print("SPEC Type\n\n", type(s))

internal = model._get_mil_internal()
if internal is not None:
    print("INTERNAL\n\n", internal)
opt_hints = model.optimization_hints
if opt_hints is not None:
    print("OPT_HINTS\n\n", opt_hints)
model.output_description


spec = model.get_spec()
layers = spec.neuralNetwork.layers  # Works for neural networks
print("LAYERS\n\n",layers)



# for layer in layers:
#     print(f"Layer: {layer.name}, Type: {layer.WhichOneof('layer')}")