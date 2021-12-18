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
# Fail on any error
set -e

# Global variables
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ARTIFACTS_DIR="${HERE}/tmp"

TF_PLATFORM="linux_amd64"
TF_VERSION="0.12.31"

#################################################
# Builds and push Docker image to GCR repository
#
# Globals:
#   - HERE
#################################################
build_docker() {
  image_tag="gcr.io/symplectic-x99/terraform:latest"

  docker build -t "${image_tag}" -f "${HERE}/Dockerfile" "${ARTIFACTS_DIR}"
  docker push "${image_tag}"
}

#################################################
# Removes build artifacts
#
# Globals:
#   - HERE
#################################################
cleanup() {
  rm -rf "${ARTIFACTS_DIR}"
}

#################################################
# Downloads Terraform binary
#
# Globals:
#   - TF_PLATFORM
#   - TF_VERSION
#################################################
prepare_terraform() {
  archive_file="terraform_${TF_VERSION}_${TF_PLATFORM}.zip"
  wget "https://releases.hashicorp.com/terraform/${TF_VERSION}/${archive_file}"
  unzip "${archive_file}" -d "${ARTIFACTS_DIR}"
  rm "${archive_file}"
}

#################################################
# Downloads google-private Terraform plugin
#
# Globals:
#   - ARTIFACTS_DIR
#   - TF_PLATFORM
#################################################
prepare_terraform_plugin() {
  archive_file="${TF_PLATFORM}.tar"
  plugins_dir="${ARTIFACTS_DIR}/.terraform.d/plugins"

  gsutil cp "gs://terraform-internal-build/$(date +%Y%m%d)/${archive_file}" \
    "${archive_file}"
  mkdir -p ${plugins_dir}
  tar xf "${archive_file}" -C "${plugins_dir}" "${TF_PLATFORM}"
  rm -rf "${archive_file}"
}

#################################################
# Script entry point
################################################
main() {
  prepare_terraform
  prepare_terraform_plugin

  build_docker

  cleanup
}

main