import os
import re
import tensorflow as tf

from src import Detector, AnchorGenerator, FeatureExtractor
from evaluation_utils import Evaluator


def show_node_name():
    print("=========================================")
    tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    for tensor_name in tensor_name_list:
        print(tensor_name)
    print("=========================================")


# model_fn输入参数features, labels就是input_fn的返回值
def model_fn(features, labels, mode, params, config):
    """This is a function for creating a computational tensorflow graph.
    The function is in format required by tf.estimator.
    """

    # the base network
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    feature_extractor = FeatureExtractor(is_training)       # facebox基础网络

    # anchor maker
    anchor_generator = AnchorGenerator()                    # 锚框生成器

    # add box/label predictors to the feature extractor
    detector = Detector(features['images'], feature_extractor, anchor_generator)    # 进行检测的实例
    # show_node_name()

    # add NMS to the graph
    if not is_training:
        # 非训练模式，进行预测
        predictions = detector.get_predictions(
            score_threshold=params['score_threshold'],
            iou_threshold=params['iou_threshold'],
            max_boxes=params['max_boxes']
        )
        # show_node_name()

    if mode == tf.estimator.ModeKeys.PREDICT:
        # this is required for exporting a savedmodel
        export_outputs = tf.estimator.export.PredictOutput({
            name: tf.identity(tensor, name)
            for name, tensor in predictions.items()
        })
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions,
            export_outputs={'outputs': export_outputs}
        )
        
    # add L2 regularization
    with tf.name_scope('weight_decay'):
        add_weight_decay(params['weight_decay'])
        regularization_loss = tf.losses.get_regularization_loss()

    # create localization and classification losses
    losses = detector.loss(labels, params)
    tf.losses.add_loss(params['localization_loss_weight'] * losses['localization_loss'])
    tf.losses.add_loss(params['classification_loss_weight'] * losses['classification_loss'])
    tf.summary.scalar('regularization_loss', regularization_loss)
    tf.summary.scalar('localization_loss', losses['localization_loss'])
    tf.summary.scalar('classification_loss', losses['classification_loss'])
    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    if mode == tf.estimator.ModeKeys.EVAL:

        filenames = features['filenames']
        batch_size = filenames.shape.as_list()[0]
        assert batch_size == 1

        with tf.name_scope('evaluator'):
            evaluator = Evaluator()
            eval_metric_ops = evaluator.get_metric_ops(filenames, labels, predictions)

        return tf.estimator.EstimatorSpec(
            mode, loss=total_loss,
            eval_metric_ops=eval_metric_ops
        )

    assert mode == tf.estimator.ModeKeys.TRAIN
    with tf.variable_scope('learning_rate'):
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.piecewise_constant(global_step, params['lr_boundaries'], params['lr_values'])
        tf.summary.scalar('learning_rate', learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops), tf.variable_scope('optimizer'):
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=True)
        grads_and_vars = optimizer.compute_gradients(total_loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)

    for g, v in grads_and_vars:
        tf.summary.histogram(v.name[:-2] + '_hist', v)
        tf.summary.histogram(v.name[:-2] + '_grad_hist', g)

    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)


def add_weight_decay(weight_decay):
    """Add L2 regularization to all (or some) trainable kernel weights."""
    weight_decay = tf.constant(
        weight_decay, tf.float32,
        [], 'weight_decay'
    )
    trainable_vars = tf.trainable_variables()
    kernels = [v for v in trainable_vars if 'weights' in v.name]
    for K in kernels:
        x = tf.multiply(weight_decay, tf.nn.l2_loss(K))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, x)
