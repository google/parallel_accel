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
# Helper script that serves ParallelAccel Client API in the SwaggerUI Docker container
HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

BASENAME="parallel_accel_client_api"
TEMPLATE_API_DOC="$(find $HERE/.. -type f -name ${BASENAME}.tpl)"
TEMPORARY_API_DOC="/tmp/${BASENAME}.yaml"

# Preprocess template file
cat $TEMPLATE_API_DOC \
    | sed 's/${title_prefix}//g' \
    | sed 's/${hostname}/REDACTED/g' \
    > $TEMPORARY_API_DOC

# Fix permissions
chmod 0644 $TEMPORARY_API_DOC

# Start SwaggerUI Docker image
echo "Starting SwaggerUI Docker container at http://127.0.0.1:8080"
echo "Please CTRL+C to stop the container"
docker run --rm -p 127.0.0.1:8080:8080 \
    -v $TEMPORARY_API_DOC:/app/api.yaml \
    -e SWAGGER_JSON=/app/api.yaml \
    swaggerapi/swagger-ui

# Clean up
rm $TEMPORARY_API_DOC