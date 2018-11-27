import tensorflow as tf
import sys

filename = 'ncsmodel/NN.pb'
#filename = 'ncsmodel/model.pb'
filename = 'ncsmodel/frozen.pb'

node_ops = []
with tf.gfile.GFile(filename, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

for node in graph_def.node:
    print(str(node.name) +  " , " + str(node.op))
