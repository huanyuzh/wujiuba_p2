from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .common import ModelBuilder
from .vggcommon import vgg_part_conv, vgg_inference, vgg_loss, vgg_eval


def original(images, labels, num_classes, total_num_examples, devices=None, is_train=True):
    """Build inference"""
    if devices is None:
        devices = [None]

    with tf.device(devices[0]):
        builder = ModelBuilder()
        net, logits, total_loss = vgg_inference(builder, images, labels, num_classes)

        if not is_train:
            return vgg_eval(net, labels)

        global_step = builder.ensure_global_step()
        # Compute gradients
        opt = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = opt.minimize(total_loss, global_step=global_step)

    return net, logits, total_loss, train_op, global_step


def ndev_data(images, labels, num_classes, total_num_examples, devices, is_train=True):
    # Your code starts here...
    return