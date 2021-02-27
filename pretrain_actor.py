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
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial
import json

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ.pop('TF_CONFIG', None)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from kerastuner.tuners import RandomSearch
from kerastuner.tuners import BayesianOptimization

from prepare_data import *

SEARCH = False


class ActorModel():

    def __init__(self, n_actions):
        self.lr = 0.005
        self.n_actions = n_actions

    def build_model(self, hp):
        input_ = layers.Input(shape=(17,))
        model = keras.Sequential()
        model.add(input_)
        model.add(layers.Dense(units=hp.Int('units_1',
                                            min_value=8,
                                            max_value=12,
                                            step=1),
                               activation='relu'))
        model.add(layers.Dense(units=hp.Int('units_2',
                                            min_value=3,
                                            max_value=6,
                                            step=1),
                               activation='relu'))
        model.add(layers.Dense(units=hp.Int('units_3',
                                            min_value=3,
                                            max_value=6,
                                            step=1),
                               activation='relu'))
        model.add(layers.Dense(self.n_actions))
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate',
                          values=[5e-2, 1e-3, 5e-3])),
            loss='mean_squared_error',
            metrics=['mean_squared_error'])
        return model

    def best_model(self):
        self.lr = 5e-3
        input_ = layers.Input(shape=(17,))
        # envs = input_[:, :6]
        # policies = input_[:, -9:-3]
        # scores = input_[:, -3:]
        model = keras.Sequential()
        model.add(input_)

        # Small 2 layer model
        # model.add(layers.Dense(units=32, activation='relu'))
        # model.add(layers.Dense(units=8, activation='relu'))

        # Bigger 3 layer model
        model.add(layers.Dense(units=32, activation='relu'))
        model.add(layers.Dense(units=20, activation='relu'))
        model.add(layers.Dense(units=8, activation='relu'))

        # Small 4 layer model
        # model.add(layers.Dense(units=12, activation='relu'))
        # model.add(layers.Dense(units=5, activation='relu'))
        # model.add(layers.Dense(units=3, activation='relu'))
        # model.add(layers.Dense(units=8, activation='relu'))

        model.add(layers.Dense(self.n_actions))
        model.compile(
            optimizer=keras.optimizers.Adam(self.lr),
            loss='mean_squared_error',
            metrics=['mean_squared_error'])
        return model

    def old_model(hp):
        input_ = layers.Input(shape=(15,))
        envs = input_[:, :6]
        policies = input_[:, -9:-3]
        scores = input_[:, -3:]
        out3 = layers.Dense(3, activation="relu")(scores)
        out3 = layers.Dense(3, activation="relu")(out3)
        out3 = layers.Dense(1, activation="relu")(out3)
        concat1 = layers.Concatenate()([policies, out3])
        out1 = layers.Dense(8, activation="relu")(concat1)
        out1 = layers.Dense(8, activation="relu")(out1)
        out1 = layers.Dense(2, activation="relu")(out1)
        concat = layers.Concatenate()([envs, out1])
        out2 = layers.Dense(128, activation="relu")(concat)
        out2 = layers.Dense(128, activation="relu")(out2)
        outputs = layers.Dense(self.n_actions)(out2)
        model = tf.keras.Model(input_, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.005),
            loss='mean_squared_error',
            # metrics=['accuracy'],
            run_eagerly=True)

        return model


def decay(epoch, lr):
    """
    Function for decaying the learning rate.
    You can define any decay function you need.
    """
    if epoch < 50:
        return lr
    return 0.1 * lr


class BatchGenerator(keras.utils.Sequence) :
  
  def __init__(self, image_filenames, labels, batch_size) :
    self.image_filenames = image_filenames
    self.labels = labels
    self.batch_size = batch_size
    
    
  def __len__(self) :
    return (np.ceil(
        len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx) :
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
    
    return np.array([
            resize(imread('/content/all_images/' + str(file_name)), (80, 80, 3))
               for file_name in batch_x])/255.0, np.array(batch_y)

class DatasetGenerator(tf.keras.utils.Sequence):
      def __init__(self, dataset, split):
          self.dataset = dataset
          self.split = split

      def __len__(self):
          return tf.data.experimental.cardinality(
            self.dataset[self.split]).numpy()

      def __getitem__(self, idx):
          return list(self.dataset[self.split].as_numpy_iterator())[idx]


def pretrain_agent():
    """
    Input: dist_to_objs(3), grip_pos(2), dist_to_obst(1) / v_agents(6) / scores(3)
    Output: v_agents[argmax(scores)]
    """
    epochs = 100
    dset = load_500_dataset()

    # tf_config = {
    #         'cluster': {
    #             'worker': ['localhost:12345', 'localhost:23456']
    #         },
    #         'task': {'type': 'worker', 'index': 0}
    #     }
    # json.dumps(tf_config)
    # num_workers = len(tf_config['cluster']['worker'])
    # strategy = tf.distribute.MultiWorkerMirroredStrategy()
    # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    training = DatasetGenerator(dset, "train")
    validation = DatasetGenerator(dset, "test")

    if SEARCH:

        tuner = BayesianOptimization(
            ActorModel(2).build_model,
            objective='val_loss',
            max_trials=20,
            executions_per_trial=3,
            directory='my_dir',
            project_name='helloworld_' + datetime.now(
                ).strftime("%Y%m%d-%H%M%S"))
        tuner.search_space_summary()
        tuner.search(dset['train'],
                     epochs=epochs,
                     validation_data=dset['test'],
                     use_multiprocessing=True,
                     workers=15)

        # Show a summary of the search
        tuner.results_summary()
    else:
        actor_model = ActorModel(2)
        model = actor_model.best_model()

        callbacks = [
            keras.callbacks.TensorBoard(
                log_dir='./tf_logs/fit/'+ datetime.now(
                    ).strftime("%Y%m%d-%H%M%S")),
            keras.callbacks.ModelCheckpoint(
                filepath='./results',
                monitor='val_loss',
                mode='auto',
                save_best_only=True),
            # keras.callbacks.EarlyStopping(
            #     monitor='val_loss', min_delta=0, patience=2, verbose=2, mode='auto',
            #     baseline=None, restore_best_weights=True),
            keras.callbacks.LearningRateScheduler(
                partial(decay, lr=actor_model.lr))
        ]

        model.fit(
            dset["train"], 
            epochs=epochs, 
            validation_data=dset["test"], 
            callbacks=callbacks,
            use_multiprocessing=True,
            workers=15
            )
            # model.save_weights('./results/model.h5')

            # for x, y in dset['test']:
            #     y_test = model.predict(x)
            #     y_test = y_test / np.linalg.norm(y_test, axis=1, keepdims=True)
            #     mse = np.mean((y - y_test)**2)
            #     print("MSE:{}".format(mse))
            #     print(y[0], y_test[0])


if __name__ == "__main__":
    pretrain_agent()
