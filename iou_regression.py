#  Copyright 2020  Marius Schubert, Tobias Riedlinger
#
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import tqdm
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
import keras
import keras.backend as ker_back
from RegscorePy import *
import matplotlib.pyplot as plt
import plotting.lasso_plots as spagh

valid_methods = ["linear", "shallow nn", "gradient boost"]


def r2_regression(variables, targets, num_ensemble=10, method="linear", nn_epochs=30,
               gb_depth=3, gb_n_estimators=100, gb_alpha=0.4, gb_lambda=0.4):
    assert method in valid_methods
    r_squared_val = []
    r_squared_train = []

    num_variables = variables.shape[-1]
    iou_predictions = []

    print("Aggregating {} regression over {} ensemble members...".format(method, num_ensemble))
    for i in tqdm.tqdm(range(num_ensemble)):
        np.random.seed(i)
        val_mask = np.random.rand(len(targets)) < 0.2
        train_mask = np.logical_not(val_mask)
        variables_val, variables_train = variables[val_mask, :], variables[train_mask, :]
        targets_val, targets_train = targets[val_mask], targets[train_mask]

        if method == "linear":
            model = LinearRegression().fit(variables_train, targets_train)
        elif method == "shallow nn":
            model = shallow_net_model(num_variables)
            model.compile(loss='mean_squared_error', optimizer='adam', metrics=[shallow_net_stddev])
            model.fit(variables_train, targets_train, epochs=nn_epochs, batch_size=128, verbose=0)
        elif method == "gradient boost":
            model = XGBRegressor(verbosity=0, max_depth=gb_depth, colsample_bytree=0.5, n_estimators=gb_n_estimators, reg_alpha=gb_alpha, reg_lambda=gb_lambda).fit(variables_train, targets_train)

        prediction = np.clip(model.predict(variables), 0, 1)
        if i == 0:
            iou_predictions.append(targets[val_mask])
            iou_predictions.append(prediction[val_mask])
        r_squared_val.append(r2_score(targets_val, prediction[val_mask]))
        r_squared_train.append(r2_score(targets_train, prediction[train_mask]))

    iou_predictions = np.array(iou_predictions)
    r_squared_val, r_squared_train = np.array(r_squared_val), np.array(r_squared_train)
    frame = pd.DataFrame({"mean R^2 val" : [np.mean(r_squared_val)],
                          "std R^2 val" : [np.std(r_squared_val)],
                          "mean R^2 train" : [np.mean(r_squared_train)],
                          "std R^2 train" : [np.std(r_squared_train)]})

    return iou_predictions, frame


def shallow_net_model(input_dim):
    reg = regularizers.l2(0.01)
    model = Sequential()
    model.add(Dense(units=61, activation='relu', kernel_regularizer=reg, input_dim=input_dim))
    model.add(Dense(units=61, activation='relu', kernel_regularizer=reg))
    model.add(Dense(units=1, kernel_regularizer=reg, activation='linear'))

    return model


def shallow_net_stddev(y_true, y_pred):
    return ker_back.sqrt(keras.losses.mean_squared_error(y_true, y_pred))


def lasso_plot(variables, targets, variables_names, ords_of_mag=[1.0e-3, 1.0e-2, 1.0e-1], scales=[1, 2, 4, 7, 9]):
    ticks = []
    weights_df = pd.DataFrame(columns=variables_names)
    information_criterion_frame = pd.DataFrame(columns=['AIC', 'BIC'])

    for ord in ords_of_mag:
        for s in tqdm.tqdm(scales):
            lam_inverse = ord * s
            ticks.append(lam_inverse)
            lasso_model = Lasso(alpha=lam_inverse, max_iter=3000, tol=1e-4).fit(variables, targets)
            weights_df.loc[lam_inverse, :] = lasso_model.coef_
            num_active_weights = int(np.count_nonzero(np.array(lasso_model.coef_)))
            if num_active_weights ==0:
                information_criterion_frame.loc[lam_inverse, 'AIC'] = 0
                information_criterion_frame.loc[lam_inverse, 'BIC'] = 0
            else:
                information_criterion_frame.loc[lam_inverse, 'AIC'] = aic.aic(targets.astype(np.float), lasso_model.predict(variables).astype(np.float), num_active_weights)
                information_criterion_frame.loc[lam_inverse, 'BIC'] = bic.bic(targets.astype(np.float), lasso_model.predict(variables).astype(np.float), num_active_weights)

    spagh.large_lasso_clean(weights_df, information_criterion_frame, ticks)

    return weights_df, information_criterion_frame

