import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
from network_helpers.frontends import resnet_v2
import os, sys
from network_helpers.models.AdapNet import build_adaptnet
from network_helpers import model_builder
# class ImportGraph():
#     """  Importing and running isolated TF graph """        
#     def __init__(self, loc):
#         # Create local graph and use it in the session          
#         self.graph = tf.Graph()
#         self.sess = tf.Session(graph=self.graph)
#         with self.graph.as_default():
#         # Import saved model from location 'loc' into local graph               
#             saver = tf.train.import_meta_graph(loc + '.meta', clear_devices=True)
#             saver.restore(self.sess, loc)
#             # There are TWO options how to get activation operation:                  
#             # FROM SAVED COLLECTION:                          
#             self.activation = tf.get_collection('activation')[0]
#             # BY NAME:                
#             self.activation = self.graph.get_operation_by_name('activation_opt').outputs[0]
#     def run(self, data):
#         """ Running the activation operation previously imported """
#         # The 'x' corresponds to name of input placeholder
#         return self.sess.run(self.activation, feed_dict={"x:0": data})

def freeze_graph(model_dir, output_node_names):
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names,
                           comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exist")

    if not output_node_names:
        print("You need to supply the name of the output node")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(args.meta_graph_path, clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the usefull nodes
        )
    return frozen_graph


def build_CMoDE(inputs, ckpt_model1, ckpt_model2, num_classes):
    """
    Builds the AdaptNet model. 

    Arguments:
      inputs: The input tensor= 
      preset_model: Which model you want to use. Select which ResNet model to use for feature extraction 
      num_classes: Number of classes

    Returns:
      CMoDE model
    """

    model1 = build_graph(ckpt_model1,inputs,num_classes,"rgb/resnet_v1_50")
    model2 = build_graph(ckpt_model2,inputs,num_classes,"rgb/resnet_v1_50")


    return model1

def build_graph(ckpt,inputs,num_classes,scope):
    model = build_adaptnet(inputs,num_classes)
    with tf.variable_scope(scope):
            model.build_graph(inputs)
    return model