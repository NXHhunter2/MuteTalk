import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

# Paths
WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
IMAGE_PATH = WORKSPACE_PATH+'/images'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'

CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 

CONFIG_PATH = MODEL_PATH+'/'+CUSTOM_MODEL_NAME+'/pipeline.config'

config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
config

# # Label map
# labels = [{'name':'Hello', 'id':1}]

# with open(ANNOTATION_PATH + '/label_map.pbtxt', 'w') as f:
#     for label in labels:
#         f.write('item { \n')
#         f.write('\tname:\'{}\'\n'.format(label['name']))
#         f.write('\tid:{}\n'.format(label['id']))
#         f.write('}\n')

# # Create directory for the custom model
# model_dir = os.path.join("Tensorflow", "workspace", "models", CUSTOM_MODEL_NAME)
# os.makedirs(model_dir, exist_ok=True)

# # Copy the pipeline config file to the custom model directory
# config_file_src = os.path.join(PRETRAINED_MODEL_PATH, "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8", "pipeline.config")
# config_file_dst = os.path.join(model_dir, "pipeline.config")
# os.system(f"cp {config_file_src} {config_file_dst}")