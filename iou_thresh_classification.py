#  Copyright 2020  Marius Schubert, Tobias Riedlinger
#
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import roc_auc_score
from RegscorePy import *
import tqdm
import plotting.lasso_plots as spagh

def classification(variables, ious, iou_threshold=0.0, num_ensemble=10):
    targets = (ious > iou_threshold)
    val_accuracy = []
    train_accuracy = []
    val_auroc = []
    train_auroc = []

    for i in tqdm.tqdm(range(num_ensemble)):
        np.random.seed(i)

        val_mask = (np.random.rand(len(targets)) < 0.2)
        train_mask = np.logical_not(val_mask)
        variables_val, targets_val = variables[val_mask, :], targets[val_mask]
        variables_train, targets_train = variables[train_mask, :], targets[train_mask]

        model = LogisticRegression(penalty='none', solver='saga', max_iter=2000, tol=1e-4).fit(variables_train, targets_train)

        predictions = model.predict(variables)
        prediction_probabilities = model.predict_proba(variables)
        val_accuracy.append(np.mean(predictions[val_mask] == targets_val))
        train_accuracy.append(np.mean(predictions[train_mask] == targets_train))
        val_auroc.append(roc_auc_score(targets[val_mask], prediction_probabilities[val_mask, 1]))
        train_auroc.append(roc_auc_score(targets[train_mask], prediction_probabilities[train_mask, 1]))

    val_accuracy, train_accuracy = np.array(val_accuracy), np.array(train_accuracy)
    frame = pd.DataFrame({"mean val acc": [np.mean(val_accuracy)],
                          "std val acc": [np.std(val_accuracy)],
                          "mean train acc": [np.mean(train_accuracy)],
                          "std train acc": [np.std(train_accuracy)],
                          "mean val auroc": [np.mean(val_auroc)],
                          "std val auroc": [np.std(val_auroc)],
                          "mean train auroc": [np.mean(train_auroc)],
                          "std train auroc": [np.std(train_auroc)]})

    return frame


def plot_classification(variables, ious, thresholds=1.0*np.arange(10)/10, num_ensemble=10,
                        error_bars=False):
    collected_frames = []
    for thresh in thresholds:
        collected_frames.append(classification(variables, ious, iou_threshold=thresh, num_ensemble=num_ensemble))

    plt.clf()
    for col in ["mean val acc", "mean train acc", "mean val auroc", "mean train auroc"]:
        plt.plot(thresholds, [df.loc[0, col] for df in collected_frames], label=col)
    plt.legend()
    plt.xlabel("IoU classification threshold")
    plt.show()
    return collected_frames


def lasso_plot(variables, ious, variables_names, iou_threshold=0.0, ords_of_mag=[1.0e-3, 1.0e-2, 1.0e-1], scales=[1, 2, 4, 7, 9]):
    targets = (ious > iou_threshold)
    ticks = []
    weights_df = pd.DataFrame(columns=variables_names)
    information_criterion_frame = pd.DataFrame(columns=['AIC', 'BIC'])

    for ord in ords_of_mag:
        for s in tqdm.tqdm(scales):
            lam_inverse = ord * s
            ticks.append(lam_inverse)
            lasso_model = LogisticRegression(penalty='l1', C=lam_inverse, solver='saga', max_iter=5000, tol=1e-4).fit(variables, targets)
            weights_df.loc[lam_inverse, :] = lasso_model.coef_
            num_active_weights = int(np.count_nonzero(np.array(lasso_model.coef_)))
            if num_active_weights ==0:
                information_criterion_frame.loc[lam_inverse, 'AIC'] = 0
                information_criterion_frame.loc[lam_inverse, 'BIC'] = 0
            else:
                information_criterion_frame.loc[lam_inverse, 'AIC'] = aic.aic(targets.astype(np.float), lasso_model.predict(variables).astype(np.float), num_active_weights)
                information_criterion_frame.loc[lam_inverse, 'BIC'] = bic.bic(targets.astype(np.float), lasso_model.predict(variables).astype(np.float), num_active_weights)

    '''
    plt.clf()
    aic_lam_inv = information_criterion_frame.index[np.argmin(information_criterion_frame['AIC'].to_numpy())]
    aic_choice = plt.plot(aic_lam_inv * np.ones([2]), np.array([weights_df.min(), weights_df.max()]), color='k')
    bic_lam_inv = information_criterion_frame.index[np.argmin(information_criterion_frame['BIC'].to_numpy())]
    bic_choice = plt.plot(bic_lam_inv * np.ones([2]), np.array([weights_df.min(), weights_df.max()]), color='k')

    graphs_list = []
    for col in weights_df.columns:
        ls = 'solid'
        if 'nab_loc' in col: ls = 'dashed'
        if 'nab_score' in col: ls = 'dashdot'
        if 'nab_prob' in col: ls = 'dotted'
        graphs_list.append(plt.plot(weights_df.index, weights_df[col], label=col, linestyle=ls))
    plt.xscale("log")
    plt.xlabel("$\lambda$")
    plt.ylabel("Regression coefficients")
    plt.title("LASSO Plot: Regression")
    plt.legend(loc="lower center")
    # plt.legend(bbox_to_anchor=(1, 0), ncol=3)
    plt.xticks(ticks=ticks, labels=ticks, rotation='vertical')
    plt.show()
    '''
    spagh.large_lasso_clean(weights_df, information_criterion_frame, ticks)

    return weights_df, information_criterion_frame
