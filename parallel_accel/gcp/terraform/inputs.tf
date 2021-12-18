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
terraform {
  experiments = [variable_validation]
}

variable "bigquery" {
  type = object({
    location  = string
    prefix = string
    project_id = string
  })

  description = <<EOT
  BigQuery dataset configuration:
    - location: Dataset location
    - prefix: Datasets prefix
    - project_id: Target Google Cloud project for dataset location
  EOT
}

variable "cluster" {
  type = object({
    location = string
    name     = string
  })

  description = <<EOT
  GKE cluster configuration:
    - name: Cluster name
    - location: Cluster location
  EOT

  validation {
    condition     = can(regex("working_area", var.cluster.name))
    error_message = "The GKE cluster name must start with the \"working_area\" prefix."
  }
}

variable "endpoints" {
  type = object({
    service_name = string
    title_prefix = string
  })
  description = <<EOT
  Cloud Endpoints service configuration:
    - service_name: Cloud Endpoints service name
    - title_prefix: Prefix to be prepended to the service title

    The service_name is the first part of the service hostname:
REDACTED
  EOT

  validation {
    condition     = can(regex("^parallel_accel", var.endpoints.service_name))
    error_message = "The service_name property must start with the \"parallel_accel\" prefix."
  }
}

variable "environment" {
  type        = string
  description = <<EOT
  Service environment:
  - development
  - production
  EOT

  validation {
    condition     = contains(["production", "staging", "development"], var.environment)
    error_message = "Allowed values for service_tier are \"production\" or \"development\"."
  }
}

variable "node_pool" {
  type = object({
    locations = list(string)
  })

  description = <<EOT
  GKE cluster configuration:
    - location: List of node pools locations.
  EOT

  validation {
    condition     = length(var.node_pool.locations) > 0
    error_message = "The node pool locations list cannot be empty."
  }
}

variable "project_id" {
  type        = string
  description = "Google Cloud Project id"
  default     = "symplectic-x99"
}
