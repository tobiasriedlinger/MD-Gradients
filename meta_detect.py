#!/usr/bin/python3
#  Copyright 2020  Marius Schubert, Tobias Riedlinger
#
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import argparse
import os
import meta_detect_settings as settings
import iou_regression as reg
import iou_thresh_classification as thresh_class
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()

    return 1

class MetaDetect(object):
    def __init__(self):
        print("Reading bounding box information...")
        self.base_frame = pd.read_csv(settings.BASE_FRAME_PATH)
        uncertainty_metrics = [self.base_frame[["s"]]]
        print("\t> Done.")
        if settings.INCLUDE_GRADIENT_METRICS:
            print("Reading gradient uncertainty metrics...")
            self.gradient_frame = pd.read_csv(settings.GRADIENT_FRAME_PATH)
            uncertainty_metrics.append(self.gradient_frame.drop(["box_id", "Unnamed: 0"], axis=1))
            print("\t> Done.")

        self.uncertainty_frame = pd.concat(uncertainty_metrics, axis=1)
        self.uncertainty_names = self.uncertainty_frame.columns
        self.standardize_data()

    def standardize_data(self):
        """
        Standardize uncertainty data such that mean(data) = 0 and std(data) = 1.
        """
        for col in self.uncertainty_frame.columns:
            dat = np.copy(np.array(self.uncertainty_frame[col]))
            self.uncertainty_frame[col] = (self.uncertainty_frame[col] - np.mean(dat)) / np.std(dat)

    def run_evaluation(self):
        """

        :return:
        """
        assert os.path.isdir(settings.METRICS_PATH)
        metrics = self.uncertainty_frame.to_numpy(copy=True)
        iou = self.base_frame["true_iou"].to_numpy(copy=True)

        if settings.RUN_REGRESSION:
            r_squared_metrics = reg.regression(metrics, iou, method="gradient boost")
            print(r_squared_metrics)

        if settings.RUN_REG_LASSO:
            weight_frame, information_criteria = reg.lasso_plot(metrics, iou, self.uncertainty_names)

        if settings.RUN_CLASSIFICATION:
            frame_list = thresh_class.plot_classification(metrics, iou, thresholds=[0.3, 0.5, 0.7, 0.9])

        if settings.RUN_CLASS_LASSO:
            weight_frame, information_criteria = thresh_class.lasso_plot(metrics, iou, self.uncertainty_names, iou_threshold=0.3)
        print("Reached End of Script!")


if __name__ == "__main__":
    options = parse_args()
    MetaDetect().run_evaluation()