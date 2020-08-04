#General Imports
import sys
import numpy as np
import pandas as pd
import argparse
import copy
import random
import json
import pickle

#Tensorflow
from absl import flags
import tensorflow as tf
from tensorflow.keras import layers

#Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from mia.estimators import ShadowModelBundle, AttackModelBundle, prepare_attack_data

from .bnlearn_data import train_input_fn, eval_input_fn 

def to_onehot(inp):
    s = pd.Series(inp)
    out = pd.get_dummies(s)
    return out
        
# Trains a simple DNN on the output probabilities of both the correlational and the causal model.
def my_attack_model(features, labels, mode, params):
    """DNN with one hidden layers and learning_rate=0.1."""
    # Create three fully connected layers.
    net = tf.feature_column.input_layer(features, params['feature_columns'])

    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
        net = tf.layers.batch_normalization(net, training=True)
        net = tf.nn.relu(net)
    
    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    #logits = logits + 0.9

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.argmax(labels,1), logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=tf.argmax(labels,1),
            predictions=predicted_classes,
            name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    # optimizer = tf.train.ProximalAdagradOptimizer(
    #     learning_rate=params['learning_rate'],
    #     l2_regularization_strength=0.001
    #   )
    optimizer = tf.train.AdamOptimizer(
            learning_rate=tf.train.exponential_decay(
            learning_rate=params['learning_rate'],
            global_step=tf.train.get_global_step(),
            decay_steps=1000,
            decay_rate=0.96)) 
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def mia(X_att_train, y_att_train, X_att_test, y_att_test, my_feature_columns, batch_size, train_steps, mdir):

    attacker = tf.estimator.Estimator(
                    model_fn = my_attack_model,
                    params={
                            'feature_columns': my_feature_columns,
                            'hidden_units': [8, 4],
                            'n_classes': 2,
                            'n_train_examples': len(X_att_train),
                            'learning_rate': 0.001
                            },
                            )
    # Train the attacker classifier
    #print(X_att_train)
    #print(y_att_train)
    attacker.train(
            input_fn=lambda:train_input_fn(X_att_train, y_att_train, batch_size), 
            steps=train_steps)

    # Evaluate the attacker model.
    eval_result_train = attacker.evaluate(input_fn=lambda:eval_input_fn(X_att_train, y_att_train, batch_size))

    # Evaluate the attacker model.
    eval_result_test = attacker.evaluate(input_fn=lambda:eval_input_fn(X_att_test, y_att_test, batch_size))

    # Get the prediction confidences from the attacker model
    predict_result = attacker.predict(input_fn=lambda:eval_input_fn(X_att_test, y_att_test, batch_size))

    attack_guess = []
    for i in predict_result:
        attack_guess.extend(i['class_ids'])

    return {'tr_attack': eval_result_train, 'te_attack': eval_result_test}  

