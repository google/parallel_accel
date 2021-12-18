#! /bin/bash
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
# Functions
usage() {
	echo "./delete_dev.sh user"
	exit
}

if [ -z "$1" ] 
then
	usage
fi

# Script
TAG=$1
ZONE="us-central1-a"

echo "Deleting VM..."
gcloud compute instances delete "parallel_accel-dev-vm-${TAG}" --zone "${ZONE}"

echo "Deleting ASIC..."
gcloud container clusters delete "parallel_accel-dev-asic-${TAG}" --zone "${ZONE}"
