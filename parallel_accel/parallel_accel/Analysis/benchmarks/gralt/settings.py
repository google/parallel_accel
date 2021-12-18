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
"""Set and load settings for benchmarks."""
import json
import os
def set_settings(
    min_subgraphs, max_subgraphs, skip_subgraphs, min_discretes, max_discretes, iterations,
    num_samples, rounding_digits, acyclic_graph_type, sim_type, full_save_dir):
  """Save a dictionary containing the settings."""
  settings_dict = {
      "min_subgraphs": min_subgraphs,
      "max_subgraphs": max_subgraphs,
      "skip_subgraphs": skip_subgraphs,
      "min_discretes": min_discretes,
      "max_discretes": max_discretes,
      "iterations": iterations,
      "num_samples": num_samples,
      "rounding_digits": rounding_digits,
      "acyclic_graph_type": acyclic_graph_type,
      "sim_type": sim_type,
  }
  settings_filename = "settings_dict.json"
  if not os.path.exists(full_save_dir):
    os.makedirs(full_save_dir)
  settings_data_file = os.path.join(full_save_dir, settings_filename)
  with open(settings_data_file, 'w') as datafile:
    json.dump(settings_dict, datafile)
def load_settings(full_save_dir):
  """Returns a dictionary with save settings."""
  settings_filename = "settings_dict.json"
  settings_data_file = os.path.join(full_save_dir, settings_filename)
  with open(settings_data_file, 'r') as datafile:
    settings_dict = json.load(datafile)
  return settings_dict