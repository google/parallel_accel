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
	echo "./create_dev.sh user"
	exit
}

gen_startup_script() {
	ASIC_IP=$1
	ASIC_NAME=$2
	ZONE=$3

	echo "#! /bin/bash" > /tmp/startup.sh
	echo "echo KUBE_GOOGLE_CLOUD_ASIC_ENDPOINTS=grpc://${ASIC_IP} >> /etc/environment" >> /tmp/startup.sh
	echo "echo ASIC_NAME=${ZONE}/${ASIC_NAME} >> /etc/environment" >> /tmp/startup.sh
	echo "echo PARALLELACCEL_FAST_BOOT=1 >> /etc/environment" >> /tmp/startup.sh
}

if [ -z "$1" ] 
then
	usage
fi

# Script
TAG=$1
ZONE="us-central1-a"

echo "Creating ASIC..."
gcloud container clusters create parallel_accel-dev-asic-"${TAG}" \
	--zone="${ZONE}" \
	--accelerator-type=v_2 \
	--version=asic_driver_nightly

ASIC_IP=$(gcloud container clusters describe "parallel_accel-dev-asic-${TAG}" --zone="${ZONE}" --format='value[separator=":"](ipAddress,port)')
gen_startup_script "${ASIC_IP}" "parallel_accel-dev-asic-${TAG}" "${ZONE}"

echo "Creating VM..."
gcloud compute instances create parallel_accel-dev-vm-"${TAG}" \
	--zone="${ZONE}" \
	--machine-type=n2-standard-4 \
	--image=ubuntu-python \
	--metadata enable-oslogin=FALSE \
	--metadata-from-file startup-script=/tmp/startup.sh \
	--service-account  remote-linear_algebra@symplectic-x99.iam.gserviceaccount.com \
	--scopes=cloud-platform

