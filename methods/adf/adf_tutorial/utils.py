import sys
from sklearn.cluster import KMeans
import joblib
import os
import tensorflow as tf
from tensorflow.python.platform import flags
from methods.adf.adf_data.bank import bank_data
from methods.adf.adf_data.census import census_data
from methods.adf.adf_data.credit import credit_data
from methods.adf.adf_utils.utils_tf import model_loss
from path import HERE

FLAGS = flags.FLAGS

datasets_dict = {'census':census_data, 'credit':credit_data, 'bank': bank_data}

def cluster(X, cluster_num=4):
    """
    Construct the K-means clustering model to increase the complexity of discrimination
    :param dataset: the name of dataset
    :param cluster_num: the number of clusters to form as well as the number of
            centroids to generate
    :return: the K_means clustering model
    """
    clf = KMeans(n_clusters=cluster_num, random_state=2019).fit(X)
    return clf

def gradient_graph(x, preds, y=None):
    """
    Construct the TF graph of gradient
    :param x: the input placeholder
    :param preds: the model's symbolic output
    :return: the gradient graph
    """
    if y == None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = tf.reduce_max(preds, 1, keepdims=True)
        y = tf.cast(tf.equal(preds, preds_max), dtype=tf.float32)
        y = tf.stop_gradient(y)
    y = y / tf.reduce_sum(y, 1, keepdims=True)

    # Compute loss
    loss = model_loss(y, preds, mean=False)

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    return grad


