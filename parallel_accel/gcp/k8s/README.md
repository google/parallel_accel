# Kubernetes configuration

This directory provides [Kustomize](https://kustomize.io/) based configuration
files for the GKE clusters.

## Install Kustomize

> **TLDR**: call [install_kustomize.sh](./scripts/install_kustomize.sh) shell
> script to get the latest version.

Although Kustomize tool is integrated into kubectl, it does not support all
strategic merge features. Hence, we need to install the latest version
manually:

```bash
curl -s "https://raw.githubusercontent.com/\
kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
mkdir bin
mv kustomize bin/
```

## Cluster credentials

> **TLDR**: call [context.sh](./scripts/context.sh) shell script to get
> cluster credentials and switch context.

First we need to obtain GKE cluster credentials and switch `kubectl` context.
Assume we want to connect to the cluster in the zone
region.

```bash
CLUSTER="CLUSTER"
CLUSTER_REGION="ZONE"

PROJECT="PROJECT"

# Obtain credentials
gcloud containers clusters get-credentials $CLUSTER --region $CLUSTER_REGION

# Switch context
CONTEXT="gke_${PROJECT}_${CLUSTER_REGION}_${CLUSTER}"
kubectl config use-context $CONTEXT

# Verify context was set correctly
if [ "$(kubectl config current-context)" != "${CONTEXT}" ]; then
    echo "kubectl context was not set correctly."
    exit 1
fi
```

## Preview configuration

Run `kubectl diff` command to preview object that will be created:

```bash
# Preview full cluster deployment
./bin/kustomize build ./overlays/development | kubectl diff -f -

# Preview API service object
./bin/kustomize build ./overlays/development/api | kubectl diff -f -
```

## Apply configuration

Run `kubectl apply` command to create or update objects:

```bash
# Create full cluster deployment
./bin/kustomize build ./overlays/development | kubectl apply -f -

# Update API service object
./bin/kustomize build ./overlays/development/api | kubectl apply -f -
```

## ASIC worker overlays

Before creating ASIC worker in the GKE cluster we need to provide deployment
specific configuration overlay file that includes worker specific API key,
its owner and the unique SHA1 label. Rather than typing this properties manually
one can use custom [script](./scripts/worker_overlay/worker_overlay.py) to
simplify this process.

Please refer to the [README.md](./scripts/worker_overlay/README.md) for detailed
usage.
