#_*_ coding:utf8 _*_
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
from keras import backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session



def _center_loss_func(features, labels, alpha, num_classes,
                      centers, feature_dim):
    assert feature_dim == features.get_shape()[1]
    label, t = tf.split(labels, 2, axis=1)
    label = K.reshape(label, [-1])
    label = tf.to_int32(label)
    #print(sess.run(labels))
    #l = tf.Variable([1, 64])

    centers_batch = tf.gather(centers, label)
    #print(sess.run(centers_batch))
    #print(sess.run(features))
    #assert tf.shape(centers_batch) == tf.shape(features)
    diff = (1 - alpha) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)
    loss = tf.reduce_mean(K.square(features - centers_batch))
    return loss

def get_center_loss(alpha, num_classes, feature_dim):
    """Center loss based on the paper "A Discriminative
       Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    # Each output layer use one independed center: scope/centers
    centers = K.zeros([num_classes, feature_dim])
    @functools.wraps(_center_loss_func)
    def center_loss(y_true, y_pred):
        return _center_loss_func(y_pred, y_true, alpha,
                                 num_classes, centers, feature_dim)
    return center_loss