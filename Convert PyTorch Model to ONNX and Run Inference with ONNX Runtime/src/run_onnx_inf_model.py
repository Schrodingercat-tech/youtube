import numpy as np 
import onnxruntime as ort    

#load onnx model 
session = ort.InferenceSession("model/iris.onnx")


test_data = np.array([
    [5.0,3.6,1.4,0.2], # 0
    [5.9,3.0,5.1,1.8], # 2
    [6.0,2.9,4.5,1.5] # 1
    ]).astype(dtype=np.float32)

# Run inference
input_name = session.get_inputs()[0].name  # Get input layer name
output_name = session.get_outputs()[0].name  # Get output layer name
results = session.run([output_name], {input_name: test_data})

# Print the result
print ("onnx inference result:")
for result in results[0]:
    print("cls : ",np.argmax(result))
