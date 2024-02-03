import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("./runs/detect/train7/weights/epoch3.onnx")  # load onnx model
tf_rep = prepare(onnx_model)  # prepare tf representation
tf_rep.export_graph("runs/detect/train7/weights/tf_model")  # export the model

# https://github.com/tensorflow/models/issues/8990
