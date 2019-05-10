import tensorflow as tf
import json
from model import model_fn


"""The purpose of this script is to export a savedmodel."""


CONFIG = 'config.json'
OUTPUT_FOLDER = 'export/run00'
GPU_TO_USE = '0'

WIDTH, HEIGHT = 1024, 1024
# size of an input image,
# set (None, None) if you want inference
# for images of variable size


tf.logging.set_verbosity('INFO')
params = json.load(open(CONFIG))
model_params = params['model_params']

config = tf.ConfigProto()
config.gpu_options.visible_device_list = GPU_TO_USE
run_config = tf.estimator.RunConfig()
run_config = run_config.replace(
    model_dir=model_params['model_dir'],
    session_config=config
)
estimator = tf.estimator.Estimator(model_fn, params=model_params, config=run_config)


# float
def serving_input_receiver_fn():
    images = tf.placeholder(dtype=tf.float32, shape=[None, HEIGHT, WIDTH, 3], name='image_tensor')
    features = {'images': images*(1.0/255.0)}
    # features = {'images': tf.to_bfloat16(images) * (1.0 / 255.0)}

    return tf.estimator.export.ServingInputReceiver(features, {'images': images})


estimator.export_savedmodel(
    OUTPUT_FOLDER, serving_input_receiver_fn
)


# quant
# def serving_input_receiver_fn_quant():
#     images = tf.placeholder(dtype=tf.float32, shape=[None, HEIGHT, WIDTH, 3], name='image_tensor')
#     features = {'images': images}
#     # features = {'images': tf.to_bfloat16(images) * (1.0 / 255.0)}
#
#     return tf.estimator.export.ServingInputReceiver(features, {'images': images})
#
#
# estimator.export_savedmodel(
#     OUTPUT_FOLDER, serving_input_receiver_fn_quant
# )
