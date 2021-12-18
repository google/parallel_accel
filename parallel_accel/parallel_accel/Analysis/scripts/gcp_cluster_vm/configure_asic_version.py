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
#! /bin/python3

# Code to set the asic driver version
import os
from cloud_asic_client import Client

c = Client(os.environ.get('ASIC_NAME').split('/')[1])
c.configure_asic_version('asic_driver0.1-dev20200911', restart_type='ifNeeded')
# c.configure_asic_version('2.4.1', restart_type='ifNeeded')
# c.configure_asic_version('asic_driver_nightly', restart_type='ifNeeded')