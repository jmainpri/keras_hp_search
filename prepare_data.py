#!/usr/bin/env python

# Copyright (c) 2021, University of Stuttgart
# All rights reserved.
#
# Permission to use, copy, modify, and distribute this software for any purpose
# with or without   fee is hereby granted, provided   that the above  copyright
# notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS  SOFTWARE INCLUDING ALL  IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR  BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR  ANY DAMAGES WHATSOEVER RESULTING  FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
# OTHER TORTIOUS ACTION,   ARISING OUT OF OR IN    CONNECTION WITH THE USE   OR
# PERFORMANCE OF THIS SOFTWARE.
#
#                                    Jim Mainprice on Wednesday February 3 2021

import os
import h5py
import numpy as np
from scipy.special import softmax
import copy

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ.pop('TF_CONFIG', None)

import tensorflow as tf


def distance_multi(p1, p2):
    return np.linalg.norm(p1 - p2, axis=-1)


def load_dictionary_from_file(directory, filename='costdata2d_10k.hdf5'):
    datasets = {}
    with h5py.File(directory + os.sep + filename, 'r') as f:
        for d in f:
            datasets[str(d)] = f[d].value
    return datasets


def env_inputs(dict_, p_grip):
    """
    static variables in dict_: t_p_objs, t_p_goal, t_p_obst
    """
    gripPos = np.array(p_grip).reshape(-1, 2)
    objPos = np.array(dict_['init_objs_pos']).reshape(-1, 2)

    dist_to_objs = [distance_multi(objPos[i], gripPos) for i in range(3)]
    dist_to_objs = np.array(dist_to_objs)
    dist_to_obst = distance_multi(dict_['obstacle_pos'], gripPos)

    input_ = np.concatenate(
        (dist_to_objs.T, gripPos, dist_to_obst.reshape(-1, 1),), axis=1)
    return input_


def tensor_flow_dataset(X, Y, test_ratio=0.2):

    X_ = tf.data.Dataset.from_tensor_slices(X)
    Y_ = tf.data.Dataset.from_tensor_slices(Y)

    d = tf.data.Dataset.zip((X_, Y_)).shuffle(1000000000)
    
    size = X.shape[0]

    test_size = int(test_ratio * size)
    dset = dict()
    dset['test'] = d.take(test_size).batch(256)
    dset['test'] = dset['test'].prefetch(1).cache()
    dset['train'] = d.skip(test_size).batch(256)
    dset['train'] = dset['train'].prefetch(1).cache()
    return dset


def load_dataset(test_ratio=0.2):

    datafile = load_dictionary_from_file(
        './results', "pretrain_dataset.hdf5")
    X_data = datafile['X']
    Y_data = datafile['Y']
    datafile1 = load_dictionary_from_file(
        './results', "pretrain_dataset1.hdf5")
    X_data1 = datafile1['X']
    Y_data1 = datafile1['Y']
    datafile2 = load_dictionary_from_file(
        './results', "pretrain_dataset2.hdf5")
    X_data2 = datafile2['X']
    Y_data2 = datafile2['Y']
    X = np.concatenate((X_data, X_data1, X_data2))
    Y = np.concatenate((Y_data, Y_data1, Y_data2))

    return tensor_flow_dataset(X, Y, test_ratio)


def load_500_hdf5():

    datafile = load_dictionary_from_file(
        './results', "pretrain_dataset_500.hdf5")
    X = datafile['X']
    Y = datafile['Y']

    # data of policies
    X_policies = X[6:12].reshape(-1, 3, 2)

    # change the scores to softmax values
    scores = X[:, -3:]
    scores = softmax(scores, axis=1)
    X = np.concatenate((X[:, :-3], scores), axis=1)

    # create v_user with target velocity + noise
    scale = 1
    Y_noise = copy.deepcopy(Y)
    noise = np.random.uniform(-1, 1, Y.shape) * scale
    Y_noise = Y_noise + noise

    X = np.concatenate((X, Y_noise), axis=1)

    return X, Y


def load_500_dataset(test_ratio=0.2):

    X, Y = load_500_hdf5()
    return tensor_flow_dataset(X, Y, test_ratio)


dset = load_dataset()
print(dset['test'])
