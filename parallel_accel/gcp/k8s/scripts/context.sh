#!/bin/bash
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
#
# Helper script for switching GKE cluster context

# Prints script usage
usage() {
  echo -e "\nUsage: $0 {working_area|working_area-staging|working_area-dev}\n"
}

case $1 in
  working_area|working_area-staging|working_area-dev) ;;
  *)
    usage;
    exit 1;
esac

CLUSTER="$1"
CLUSTER_REGION="us-central1"
PROJECT="symplectic-x99"
if [ "$CLUSTER" == "working_area" ]; then
  PROJECT="sandbox-at-alphabet-parallel_accel-prod"
fi

while true; do
  read -p "Are you sure you want to switch context to the $PROJECT:$CLUSTER? " yn
  case $yn in
    [Yy]* ) break;;
    [Nn]* ) exit;;
    * ) echo "Please answer yes or no.";;
  esac
done

# Obtain credentials

gcloud container clusters get-credentials $CLUSTER --region=$CLUSTER_REGION \
  --project=$PROJECT

# Switch context
CONTEXT="gke_${PROJECT}_${CLUSTER_REGION}_${CLUSTER}"
kubectl config use-context $CONTEXT

# Verify context was set correctly
CURRENT_CONTEXT="$(kubectl config current-context)"
if [ "$CURRENT_CONTEXT" != "$CONTEXT" ]; then
  echo "kubectl context was not set correctly."
  exit 1
fi