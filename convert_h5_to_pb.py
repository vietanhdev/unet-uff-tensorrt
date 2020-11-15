import numpy as np
import tensorflow as tf 
import segmentation_models as sm
import importlib
import glob
# from tensorflow.keras.metrics import TrueNegatives, TruePositives, FalseNegatives, FalsePositives
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping, LearningRateScheduler, TensorBoard
from tensorflow.graph_util import convert_variables_to_constants, remove_training_nodes
from segmentation_models import Unet
from segmentation_models.metrics import IOUScore, FScore, Precision, Recall

from model.datasets import Dataset, Dataloader, get_training_augmentation, get_validation_augmentation, visualize

H5_MODEL = "saved_models/config13/model_.112-0.386539.h5"
OUTPUT_PB_FILE = "lane_segmentation_384x384.pb"

# Reload the model and the best weights
tf.keras.backend.clear_session()
tf.keras.backend.set_learning_phase(False)
model = tf.keras.models.load_model(H5_MODEL, compile=False)
# model.layers.pop()
# input = model.input
# last_layer = 'final_conv'
# output = (model.get_layer(name=last_layer).output if isinstance(last_layer, str)
#      else model.get_layer(index=last_layer).output)
# model = tf.keras.Model(inputs=input, outputs=output)
model.summary()

# Freeze model and save
# First freeze the graph and remove training nodes.
sess = tf.keras.backend.get_session()
input_graph_def = sess.graph.as_graph_def()
output_names = model.output.op.name
print(output_names)
frozen_graph = convert_variables_to_constants(sess, input_graph_def, [output_names])
frozen_graph = remove_training_nodes(frozen_graph)
# Save the model
with tf.gfile.GFile(OUTPUT_PB_FILE, "wb") as ofile:
    ofile.write(frozen_graph.SerializeToString())
print("saved model to {}".format(OUTPUT_PB_FILE))