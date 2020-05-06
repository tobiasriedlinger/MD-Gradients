#  Copyright 2020  Marius Schubert, Tobias Riedlinger
#
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import json

STATE_DICT_NAME    = "test_state"

METRICS_PATH       = "/home/riedlinger/tensorflow-yolov3-MetaDetect/metric-inferences"

DATASET_NAME       = "coco"

BASE_FRAME_FILE    = "boxes_data.csv"
BASE_FRAME_PATH    = "{}/{}/{}".format(METRICS_PATH, "COCO", BASE_FRAME_FILE)

INCLUDE_GRADIENT_METRICS = True

GRADIENT_FRAME_FILE= "50images_last_layers.csv"
GRADIENT_FRAME_PATH= "{}/{}/gradient_based/{}".format(METRICS_PATH, "COCO", GRADIENT_FRAME_FILE)

### PLOTTING:
CREATE_PLOTS       = True
USE_LATEX          = False

state_dict = {     'metrics path'   : METRICS_PATH,
                   'dataset name'   : DATASET_NAME,
                   'base frame file': BASE_FRAME_FILE,
                   'base frame path': BASE_FRAME_PATH,
                   'include grads'  : INCLUDE_GRADIENT_METRICS,
                   'grad frame file': GRADIENT_FRAME_FILE,
                   'grad frame path': GRADIENT_FRAME_PATH,
                   'create plots'   : CREATE_PLOTS,
                   'use latex'      : USE_LATEX }

json_path = "./attribute_dicts/{}.json".format(STATE_DICT_NAME)
print("Saving current attributes specified in save_state_dict.py to ...".format(json_path))

with open(json_path, 'w') as file:
    json.dump(state_dict, file)
print("... Finished!")