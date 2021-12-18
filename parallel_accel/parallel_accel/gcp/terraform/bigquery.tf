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
# Create BigQuery dataset for ASIC deployments log
resource "google_bigquery_dataset" "log_sink" {
  dataset_id  = "${replace(var.cluster.name, "-", "_")}_logs"
  description = <<EOT
  Dataset containing all logs from the working_area-api and working_area-asic deployments running in the ${var.cluster.name} GKE cluster.
  ${local.description_suffix}
  EOT

  location                        = var.bigquery.location
  project                         = var.bigquery.project_id
  default_partition_expiration_ms = 31536000000 # One year
}

# Create BigQuery log sink for ASIC deployments
resource "google_logging_project_sink" "default" {
  name = "${local.name_prefix}-working_area-sink"

  destination = "REDACTED
  filter      = <<EOT
  resource.type="k8s_container"
  resource.labels.project_id="${var.project_id}"
  resource.labels.location="${var.cluster.location}"
  resource.labels.cluster_name="${var.cluster.name}"
  (labels.k8s-asic_cluster/app=~"^working_area-.*" OR labels.k8s-asic_cluster/job-name=~"^working_area-.*")
  EOT

  bigquery_options {
    use_partitioned_tables = true
  }

  unique_writer_identity = true
}

# Grant sink service account BQ dataset editor role
resource "google_bigquery_dataset_iam_member" "default" {
  project    = var.bigquery.project_id

  dataset_id = google_bigquery_dataset.log_sink.dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = google_logging_project_sink.default.writer_identity
}

# Create BigQuery dataset for API service events
resource "google_bigquery_dataset" "api_service_events" {
  dataset_id  = "${var.bigquery.prefix}ApiServiceEvents"
  description = <<EOT
  Dataset containing ${var.cluster.name} GKE cluster API service events.
  ${local.description_suffix}
  EOT

  project                         = var.bigquery.project_id
  location                        = var.bigquery.location
  default_partition_expiration_ms = 31536000000 # One year
}

# Create BigQuery dataset for ASIC events
resource "google_bigquery_dataset" "asic_worker_events" {
  dataset_id  = "${var.bigquery.prefix}AsicWorkerEvents"
  description = <<EOT
  Dataset containing ${var.cluster.name} GKE cluster ASIC workers events.
  ${local.description_suffix}
  EOT

  project                         = var.bigquery.project_id
  location                        = var.bigquery.location
  default_partition_expiration_ms = 31536000000 # One year
}
