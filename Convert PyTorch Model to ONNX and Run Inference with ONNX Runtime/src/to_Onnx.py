import onnxruntime
from TorchModel import model,X_train
import torch 


net = model.load_state_dict(torch.load("model/iris.pth"))
model.eval()

dummy_input  = torch.randn(1,4)

# set dymaic_axes for dynamic batch sizes
torch.onnx.export(model, dummy_input, 'model/iris.onnx', input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

print ('onnx model saved')