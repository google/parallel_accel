# Copyright 2021 The ParallelAccel Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


styles = { "axes.titlesize": 28,
           "axes.labelsize": 20,
           "axes.titlepad": 20,
           "lines.linewidth": 1,
           "lines.markersize": 20,
           "xtick.labelsize": 10,
           "ytick.labelsize": 10,
           "ytick.major.size": 7,
           "ytick.minor.size": 5,
           "ytick.major.width": 2,
           "ytick.minor.width": 1,
           "ytick.direction": 'in',
           "ytick.color": 'grey',
           "ytick.labelcolor": 'black',
           "figure.figsize": (16, 12),
           "font.size": 16,
           "legend.fontsize": 16,
           "figure.autolayout": True,
}
LINESTYLES = ['solid',                  # solid
              'dashed',              # dashed
              'dotted',              # dotted
              'dashdot',        # dashdotted
              (0, (3, 5, 1, 5, 1, 5)),  # dashdotdotted
              (0, (5, 1)),              # densely dashed
              (0, (1, 1)),              # densely dotted
              (0, (3, 1, 1, 1)),        # densely dashdotted
              (0, (3, 1, 1, 1, 1, 1)),  # densely dashdotdotted
              (0, (5, 10)),             # loosely dashed
              (0, (1, 10)),             # loosely dotted
              (0, (3, 10, 1, 10)),      # loosely dashdotted
              (0, (3, 10, 1, 10, 1, 10))]# loosely dashdotdotted
MARKERS = Line2D.filled_markers

plt.style.use(['seaborn', styles])
