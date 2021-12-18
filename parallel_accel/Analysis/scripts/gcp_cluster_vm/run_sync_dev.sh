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
	echo "./run_sync_dev.sh user"
	exit
}

gen_lsyncd_config() {
USER=$1
HOST=$2
cat > /tmp/lsyncd.config << EOF
	settings {
		nodaemon = true,
	}

	sync {
		default.rsync,
		source="../../..",
		target="$USER@$HOST:parallel_accel",
		delay=1,
		rsync = {
			archive = true,
			compress = true,
			whole_file = false,
			rsh = "ssh -i ~/.ssh/google_compute_engine",
			verbose = true,
			_extra = {"--exclude-from=../../../.dockerignore"}
		},
	}
EOF
}

if [ -z "$1" ] 
then
	usage
fi


# Script
TAG=$1
ZONE=us-central1-a
VM_IP=$(gcloud compute instances describe "parallel_accel-dev-vm-${TAG}" --zone="${ZONE}" \
--format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo "Generating config..."
gen_lsyncd_config $TAG $VM_IP

echo "Starting file sync in the background..."
lsyncd /tmp/lsyncd.config

# echo "Starting jupyter..."
# ssh -i ~/.ssh/google_compute_engine -o StrictHostKeyChecking=no -L 8888:localhost:8888  -t "${TAG}"@"${VM_IP}" "jupyter notebook"
