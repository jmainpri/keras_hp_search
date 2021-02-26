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
import matplotlib.pyplot as plt


def plot(env_dict, u_opt, show_plot=True, **kwargs):

    fig, ax = plt.subplots(
                subplot_kw={'xticks': [], 'yticks': []}, )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect(aspect='equal', share=True)

    # p_objs = np.array(env_dict['init_objs_pos']).reshape(-1,2)
    # for x,y in p_objs: ax.scatter(x, y, color='g' )

    colors = ['r', 'g', 'b']
    for i in range(3):
        ax.scatter(
            env_dict['goal_pos_' + str(i)][0],
            env_dict['goal_pos_' + str(i)][1], 
            color=colors[i])
    ax.scatter(
            env_dict['obstacle_pos'][0],
            env_dict['obstacle_pos'][1], 
            color='k')

    x = env_dict['gripPos'][0]
    y = env_dict['gripPos'][1]

    for i in range(3):
        dx = env_dict['subpolicy_' + str(i)][0]
        dy = env_dict['subpolicy_' + str(i)][1]
        v = np.array([dx, dy])
        v /= np.linalg.norm(v)
        v *= .2
        ax.arrow(x, y, v[0], v[1], width=0.003, color=colors[i])

    ax.arrow(x, y, u_opt[0], u_opt[1], color='grey')

    if not show_plot: return fig, ax
    if 'figname' in kwargs: fig.savefig(kwargs['figname'])
    else:
        fig.show()
        plt.pause(60)



if __name__ == "__main__":
    datasets = {}
    with h5py.File("results/pretrain_dataset_500.hdf5", 'r') as f:
        # dist_to_objs[3]/gripPos[2]/dist_to_obst[1]/v_agents[6]/scores[3]
        np.random.seed(0)
        for j in range(1000):
            i = np.random.randint(0, 10000)
            X = f['X'][i, :]
            Y = f['Y'][i, :]
            env_dict = dict()
            env_dict['dist_to_objs_0'] = X[0]
            env_dict['dist_to_objs_1'] = X[1]
            env_dict['dist_to_objs_2'] = X[2]
            env_dict['gripPos'] = X[3:5]
            env_dict['dist_to_obst'] = X[6]
            env_dict['subpolicy_0'] = X[6:8]
            env_dict['subpolicy_1'] = X[8:10]
            env_dict['subpolicy_2'] = X[10:12]
            env_dict['score_0'] = X[12]
            env_dict['score_1'] = X[13]
            env_dict['score_2'] = X[14]
            env_dict['goal_pos_0'] = f['env_dict'][i, 0:2]
            env_dict['goal_pos_1'] = f['env_dict'][i, 2:4]
            env_dict['goal_pos_2'] = f['env_dict'][i, 4:6]
            env_dict['obstacle_pos'] = f['env_dict'][i, 6:8]
            print("----- {} -----".format(i))
            print("score_r : {}".format(env_dict['score_0']))
            print("score_g : {}".format(env_dict['score_1']))
            print("score_b : {}".format(env_dict['score_2']))
            plot(env_dict, Y)
