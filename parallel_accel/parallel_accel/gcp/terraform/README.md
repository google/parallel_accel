# Google Cloud infrastructure configuration

This directory provides [Terraform](https://www.terraform.io/) configuration
for the Google Cloud infrastructure.

## Input variables

All input variables are described in the [inputs.tf](./inputs.tf) resource file.

- **cluster** variable defines GKE cluster configuration:
  - **name** - a unique cluster name that starts with **working_area** prefix.
  - **location** - location of the cluster.
- **node_pool** variable defines GKE cluster node pool configuration:
  - **locations** - list of node pools locations.
- **endpoints** variable defines Cloud Endpoints deployed service properties:
  - **service_name** is the first part of the service hostname:
    `{service_name}.REDACTED`
  - **title_prefix** is the string text prepended to the service title.
- **environment** - service environment, either **development** or
  **production**.

For the convinence there are pre-configured settings for the different project
environments:

- [development](./envs/development.tfvars)
- [production](./envs/production.tfvars)

## Workspace

In our project we are using the same configuration code for Google Cloud
infrastructure. This means making any changes to the files may affect both
services running in the development and production environment. Terraform
provides so called
[workspaces](https://www.terraform.io/docs/language/state/workspaces.html) to
easily switch between different backends configurations.

To initialize project environment specific workspace run following shell
commands:

```bash

# Create a new terraform workspace
terraform workspace new WORKSPACE_NAME

# Switch to the new workspace
terraform workspace select WORKSPACE_NAME
```

## Deployment

> WARN: ALWAYS preview infrastructure changes before deploying them to the
> Google Cloud.

The infrastructure deployment process is composed of two stages: planning and
deploying.

In the planning stage Terraform processes and evaluates the input
configuration file. The local state is compared against state file stored in the
REDACTED Google Cloud Storage bucket. The difference is
the Terraform deployment plan.

In the deployment stage Terraform makes actual changes to the Google Cloud
infrastructure according to the input plan.

Shell commands below are the example of applying changes to the development
environment.

```bash
ENVIRONMENT=development

# Initialize local .terraform directory
terraform init

# Preview changes and save plan to the local file
terraform plan -var-file=./envs/$ENVIRONMENT.tfvars -out=$ENVIRONMENT.tfplan

# Apply new changes
terraform apply "${ENVIRONMENT}.tfplan"
```

## Configuration development

Before modifying Terraform configuration, first we need to update
[backend.tf](./backend.tf) to keep state file locally instead of GCS bucket.

```tf
# DO NOT SUBMIT
terraform {
  backend "local" {
    path = "terraform.tfstate"
  }
}
```

Now call `terraform init` and accept the suggestion to copy state from the
remote bucket to your work directory. Once done making changes to the Terraform,
revert backend settings in [backend.tf](./backend.tf) file and send changes for
the review. Finally, remove the local state file.

```bash
rm terraform.tfstate*
```
