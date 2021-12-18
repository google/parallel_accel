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
# Local variables
locals {
  # DNS name for the Cloud Endpoints service
  dns_name = "${local.hostname}."
  # Cloud Endpoints service name
REDACTED"
  # GKE node pool service account roles
  roles = [
    "cloudasic.serviceAgent",
    "cloudtrace.agent",
    "container.serviceAgent",
    "logging.logWriter",
    "monitoring.metricWriter",
    "monitoring.viewer",
    "servicemanagement.serviceController",
    "stackdriver.resourceMetadata.writer",
    "storage.objectViewer"
  ]

  # Resource name prefix
  name_prefix = "tf-${lower(var.environment)}-${substr(sha1(var.cluster.name), 0, 8)}"
  # Resource description suffix
  description_suffix = <<EOT

  This resource is managed by Terraform, do not edit it directly.
  EOT

  # working_area-api service port
  # See more: ProjectRoot/gcp/k8s/base/api/service.yaml
  service_port = 8081
}

# Create Google managed SSL certificate
resource "google_compute_managed_ssl_certificate" "default" {
  provider = google-private

  name        = "${local.name_prefix}-ssl-cert"
  description = <<EOT
  SSL certificate for the ${local.hostname} domain.
  ${local.description_suffix}
  EOT

  type = "MANAGED"
  managed {
    domains = [local.dns_name]
  }
}

# Create global external IP address
resource "google_compute_global_address" "default" {
  name        = "${local.name_prefix}-ip-address"
  description = <<EOT
  Global IP address for the ${local.hostname} hostname.
  ${local.description_suffix}
  EOT

  address_type = "EXTERNAL"
  ip_version   = "IPV4"
}

# Create DNS managed zone for Cloud Endpoints service
resource "google_dns_managed_zone" "default" {
  name        = "${local.name_prefix}-dns-zone"
  description = <<EOT
  DNS managed zone for the ${local.dns_name}
  ${local.description_suffix}
  EOT

  dns_name = local.dns_name
}

# Create DNS record
resource "google_dns_record_set" "default" {
  name         = local.dns_name
  type         = "A"
  ttl          = 3600
  managed_zone = google_dns_managed_zone.default.name
  rrdatas      = [google_compute_global_address.default.address]
}

# Create Cloud Endpoints service
resource "google_endpoints_service" "default" {
  service_name = local.hostname
  project      = var.project_id
  openapi_config = templatefile("./templates/parallel_accel_client_api.tpl", {
    hostname     = local.hostname
    ip_address   = google_compute_global_address.default.address
    title_prefix = var.endpoints.title_prefix
  })
}

# Create GKE cluster
resource "google_container_cluster" "default" {
  provider = google-private

  name        = var.cluster.name
  description = <<EOT
  GKE cluster for the ${var.environment} service tier.
  ${local.description_suffix}
  EOT

  location = var.cluster.location

  # We want to provide our custom node pool configuration, but we cannot create
  # cluster without one. Instead create with only one node and remove it right
  # after setting up the cluster.
  initial_node_count       = 1
  remove_default_node_pool = true

  ip_allocation_policy {
    # We need to enable IP aliasing (required by ASIC) for our cluster.
  }

  # Features to be enabled in the cluster
  enable_asic = true
}

# Create node pool default service account
resource "google_service_account" "default" {
  display_name = "Service account for the ${var.cluster.name} cluster"
  description  = <<EOT
  Default service account for the ${var.cluster.name} cluster node pool.
  ${local.description_suffix}
  EOT

  account_id = "${var.cluster.name}-sa"
}

# Assign roles to the service account
resource "google_project_iam_member" "default" {
  for_each = toset(local.roles)
  role     = "roles/${each.value}"
  member   = "serviceAccount:${google_service_account.default.email}"

  depends_on = [
    google_service_account.default
  ]
}

# Docker images live in symplectic-x99 project, but the production environment is
# running in sandbox-at-alphabet-parallel_accel-prod project. We need to provide GCR read
# access for the service account to be able to pull images.
resource "google_project_iam_member" "gcr_viewer" {
  count   = var.environment == "production" ? 1 : 0

  project  = "symplectic-x99"
  role     = "roles/storage.objectViewer"
  member   = "serviceAccount:${google_service_account.default.email}"

  depends_on = [
    google_service_account.default
  ]
}

# Create custom node pool for the GKE cluster
resource "google_container_node_pool" "default" {
  provider = google-private

  name = "default-pool"

  cluster  = google_container_cluster.default.name
  location = var.cluster.location

  autoscaling {
    min_node_count = 1
    max_node_count = 6
  }

  node_config {
    image_type   = "COS_CONTAINERD"
    machine_type = "e2-standard-16"

    metadata = {
      disable-legacy-endpoints = "true"
    }

    service_account = google_service_account.default.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    # Apply custom tag that will be used for the firewall rule
    tags = ["${var.cluster.name}-node-pool"]
  }

  node_locations = var.node_pool.locations
}

# Create health check
resource "google_compute_health_check" "default" {
  name        = "${local.name_prefix}-hc"
  description = <<EOT
  Default HTTP health check for the working_area-api backend service.
  ${local.description_suffix}
  EOT

  timeout_sec         = 15
  check_interval_sec  = 15
  healthy_threshold   = 1
  unhealthy_threshold = 3

  http_health_check {
    port         = local.service_port
    request_path = "/health"
  }
}

# Create firewall rules for the health check.
# See more: https://cloud.google.com/kubernetes-engine/docs/how-to/standalone-neg#attaching-ext-https-lb
resource "google_compute_firewall" "default" {
  name        = "${local.name_prefix}-fw-rule"
  description = <<EOT
  GCE L7 health check and proxy firewall rule for the ${google_compute_backend_service.default.name} backend service.
  ${local.description_suffix}
  EOT

  network = "default"

  direction = "INGRESS"
  allow {
    protocol = "tcp"
    ports    = [local.service_port]
  }

  source_ranges = ["130.211.0.0/22", "35.191.0.0/16"]

  # Apply custom tag that will be used for the firewall rule
  target_tags = ["${var.cluster.name}-node-pool"]
}

# Create NEGs
resource "google_compute_network_endpoint_group" "default" {
  for_each = toset(var.node_pool.locations)

  name        = "${var.cluster.name}-neg"
  description = <<EOT
  Network Endpoint Group for the working_area-api service running in the ${var.cluster.name} cluster.
  ${local.description_suffix}
  EOT

  network      = "default"
  subnetwork   = "default"
  default_port = local.service_port
  zone         = each.value
}

# Create backend service
resource "google_compute_backend_service" "default" {
  name        = "${local.name_prefix}-backend-service"
  description = <<EOT
  Backend service for the ${var.cluster.name} NEGs.
  ${local.description_suffix}
  EOT

  port_name   = "http"
  protocol    = "HTTP"
  timeout_sec = 620

  health_checks = [google_compute_health_check.default.id]

  dynamic "backend" {
    for_each = toset([for neg in google_compute_network_endpoint_group.default : neg.id])

    content {
      group                 = backend.value
      balancing_mode        = "RATE"
      max_rate_per_endpoint = 8
    }
  }

  log_config {
    enable      = true
    sample_rate = 0.1
  }

  depends_on = [
    google_container_node_pool.default
  ]
}

# Create frontend URL map
resource "google_compute_url_map" "default" {
  name        = "${local.name_prefix}-url-map"
  description = <<EOT
  URL map for the ${local.hostname} hostname.
  ${local.description_suffix}
  EOT

  default_service = google_compute_backend_service.default.id

  host_rule {
    hosts        = [local.hostname]
    path_matcher = "allpaths"
  }

  path_matcher {
    name            = "allpaths"
    default_service = google_compute_backend_service.default.id
  }
}

# Create frontend HTTPS proxy
resource "google_compute_target_https_proxy" "default" {
  name        = "${local.name_prefix}-https-proxy"
  description = <<EOT
  HTTPS target proxy for the ${google_compute_backend_service.default.name} backend service.
  ${local.description_suffix}
  EOT

  url_map          = google_compute_url_map.default.id
  ssl_certificates = [google_compute_managed_ssl_certificate.default.id]
}

# Create frontend HTTPS forwarding rule
resource "google_compute_global_forwarding_rule" "default" {
  name        = "${local.name_prefix}-forwarding-rule"
  description = <<EOT
  Forwarding rule for the ${google_compute_target_https_proxy.default.name} HTTPS proxy.
  ${local.description_suffix}
  EOT

  load_balancing_scheme = "EXTERNAL"
  target                = google_compute_target_https_proxy.default.id

  ip_address = google_compute_global_address.default.address
  port_range = 443
}
