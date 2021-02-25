import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
#from kerastuner.tuners import RandomSearch


def distance_multi(p1, p2):
    return np.linalg.norm(p1 - p2, axis=-1)


def load_dictionary_from_file(directory, filename='costdata2d_10k.hdf5'):
    datasets = {}
    with h5py.File(directory + os.sep + filename, 'r') as f:
        for d in f:
            datasets[str(d)] = f[d].value
    return datasets


class ActorModel():

    def __init__(self, n_actions):
        self.n_actions = n_actions

    def build_model(hp):
        input_ = layers.Input(shape=(15,))
        envs = input_[:, :6]
        policies = input_[:, -9:-3]
        scores = input_[:, -3:]
        model = keras.Sequential()
        model.add(input_)
        model.add(layers.Dense(units=hp.Int('units',
                                            min_value=32,
                                            max_value=512,
                                            step=32),
                               activation='relu'))
        model.add(layers.Dense(self.n_actions))
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate',
                          values=[1e-2, 1e-3, 1e-4])),
                loss='mean_squared_error',
                metrics=['accuracy'])
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


def pretrain_agent():
    """
    Input: dist_to_objs(3), grip_pos(2), dist_to_obst(1) / v_agents(6) / scores(3)
    Output: v_agents[argmax(scores)]
    """

    datafile = load_dictionary_from_file('./results', "pretrain_dataset.hdf5")
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

    size = X.shape[0]
    X_ = tf.data.Dataset.from_tensor_slices(X)
    Y_ = tf.data.Dataset.from_tensor_slices(Y)

    d = tf.data.Dataset.zip((X_, Y_)).shuffle(1000000000)

    epochs = 60
    test_ratio = 0.2
    test_size = int(test_ratio * size)
    dset = dict()
    dset['test'] = d.take(test_size).batch(256)
    dset['test'] = dset['test'].prefetch(tf.data.experimental.AUTOTUNE)
    dset['train'] = d.skip(test_size).batch(256)
    dset['train'] = dset['train'].prefetch(tf.data.experimental.AUTOTUNE)

    # these parameters not used here
    Actor = ActorModel(2)
    model = Actor.model

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath='./results',
        monitor='val_loss',
        mode='auto',
        save_best_only=True)
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=2, verbose=2, mode='auto',
        baseline=None, restore_best_weights=True)

    tuner = RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=5,
        executions_per_trial=3,
        directory='my_dir',
        project_name='helloworld')

    # model.fit(dset['train'], epochs=epochs, validation_data=dset[
    #           'test'], callbacks=[checkpoint, early_stopping])
    # model.save_weights('/.model.h5')

    # for x, y in dset['test']:
    #     y_test = model.predict(x)
    #     y_test = y_test / np.linalg.norm(y_test, axis=1, keepdims=True)
    #     mse = np.mean((y - y_test)**2)
    #     print("MSE:{}".format(mse))
    #     print(y[0], y_test[0])

if __name__ == "__main__":
    pretrain_agent()
