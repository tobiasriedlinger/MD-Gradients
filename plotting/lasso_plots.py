#  Copyright 2020  Marius Schubert, Tobias Riedlinger
#
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def lasso_spaghetti(weights_df, information_criterion_frame, ticks):
    plt.clf()
    tick_labels = [":.4f".format(lam) for lam in ticks]
    num_plots = len(weights_df.columns)
    grid_size = np.ceil(np.sqrt(num_plots))

    plt.style.use("seaborn-darkgrid")
    palette = plt.get_cmap("Set1")
    num = 0
    for col in weights_df.columns:
        num += 1
        plt.subplot(grid_size, grid_size, num)

        for metric in weights_df.columns:
            plt.plot(weights_df.index, weights_df[metric], marker="", color="grey", linewidth=0.6, alpha=0.3)

        plt.plot(ticks, weights_df[col], marker="", color=palette(num), linewidth=2.4, alpha=0.9, label=col)

        plt.xlim(np.min(ticks), np.max(ticks))
        plt.ylim(-5, 5)
        plt.xscale('log')
        plt.xticks(ticks=ticks, labels=tick_labels, rotation='vertical')

        plt.title(col, loc='left', fontsize=12, fontweight=0, color=palette(num))

    plt.suptitle("LASSO Plot", fontsize=13, fontweight=0, color='black', style='italic')

    plt.text(0.5, 0.2, "Penalty Strength $\\lambda$", ha='center', va='center')
    plt.text(0.06, 0.5, "Weight Size", ha='center', va='center', rotation='vertical')
    plt.show()

    return 1


def large_lasso_clean(weights_df, information_criterion_frame, ticks, color_thresh=0.3):
    plt.clf()
    tick_labels = ["{:.4f}".format(lam) for lam in ticks]
    palette = plt.get_cmap("Set1")

    aic_marker = information_criterion_frame.index[np.argmin(information_criterion_frame['AIC'].to_numpy())]
    bic_marker = information_criterion_frame.index[np.argmin(information_criterion_frame['BIC'].to_numpy())]

    for col in weights_df.columns:
        plt.plot(weights_df.index, weights_df[col], color='grey', linewidth=0.5, alpha=0.3)

    num = 0
    for col in weights_df.columns:
        if np.max(np.abs(weights_df[col].to_numpy())) > color_thresh:
            num += 1
            plt.plot(weights_df.index, weights_df[col], color=palette(num), linewidth=1.5, alpha=0.9, label=col)

    plt.xscale('log')
    plt.xlabel("Regularization Strength $\\lambda$")
    plt.ylabel("Weight Size")
    plt.legend(loc='lower center', ncol=3)
    plt.xticks(ticks=ticks, labels=tick_labels, rotation='vertical')
    plt.show()

    return 1


