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


def plot_LWR(env_dict, show_plot=True, **kwargs):
    
    fig, ax = plt.subplots(
                subplot_kw={'xticks': [], 'yticks': []}, )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect(aspect='equal', share=True)

    p_objs = np.array(env_dict['init_objs_pos']).reshape(-1,2)
    for x,y in p_objs: ax.scatter(x, y, color='g' )
    
    colors = ['r', 'g', 'b']
    for i in range(3):
        ax.scatter(
            env_dict['obstacle_pos_' + str(i)][0],
            env_dict['obstacle_pos_' + str(i)][1], 
            color=colors[i] )

    for i in range(3):
        ax.arrow(x, y, dx, dy, color=colors[i])

    if not show_plot: return fig, ax
    if 'figname' in kwargs: fig.savefig(kwargs['figname'])
    else:
        fig.show()
        plt.pause(60)


if __name__ == "__main__":
    datasets = {}
    with h5py.File("results/pretrain_dataset_500.hdf5", 'r') as f:
        X = f['X'][0, :]
        env_dict = dict()
        env_dict['obstacle_pos_0']=X[0]
