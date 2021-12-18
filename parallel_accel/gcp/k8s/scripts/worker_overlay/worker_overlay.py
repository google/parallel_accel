#!env/bin/python
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
"""A helper script to generate ASIC Worker overlay configuration file."""
import argparse
import base64
import os
import hashlib
import sys
from typing import Optional
import jinja2


HERE = os.path.dirname(os.path.realpath(__file__))
K8S_PATH = os.path.join(HERE, "../..")
OVERLAYS_PATH = os.path.join(K8S_PATH, "overlays/workers")


def compute_hash(api_key: str) -> str:
    """Computes SHA1 hash over input API key.

    Args:
        api_key: Key to be hashed.

    Returns:
        Hashed API key.
    """
    hasher = hashlib.sha1()
    hasher.update(api_key.encode())
    return hasher.hexdigest()


def generate(
    api_key: str, label: str, image_version: Optional[str] = None
) -> str:
    """Generate overlay config files.

    Args:
        api_key: API key to be assigned to the worker.
        label: API key owner.
        image_version: Optional Docker image version.
    """
    dirname = f"working_area-asic-{label}"
    dir_path = os.path.join(OVERLAYS_PATH, dirname)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    properties = {
        "api_key": base64.encodebytes(api_key.encode()).decode(),
        "api_key_hash": compute_hash(api_key),
        "image_version": image_version,
        "label": label,
    }

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(HERE),
    )

    for template_file, output_file in (
        ("worker_secret.tpl", "secret.yaml"),
        ("worker_overlay.tpl", "kustomization.yaml"),
    ):
        template = env.get_template(template_file)
        content = template.render(properties=properties)

        with open(os.path.join(dir_path, output_file), "w+") as output:
            output.write(content)

    return dirname


def main() -> None:
    """Script entry point."""
    parser = argparse.ArgumentParser(
        "ASIC Worker overlay generator",
        description=(
            "Generates kustomization.yaml overlay file for the ASIC worker"
            " deployment schema using provided input arguments. By default the"
            " worker is running the latest available version of the simulator"
            " app Docker image, which can be overriden by --image-version input"
            " argument."
        ),
        epilog=(
            "Number of --api-key arguments must be the same as number of"
            " --label arguments."
        ),
    )
    parser.add_argument(
        "--api-key",
        help="API key to be bound to the ASIC worker",
        type=str,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--label",
        help="Custom label to be appended to the deployment name",
        type=str,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--image-version",
        help="Optional simulator application version (EXAMPLE: \"staging\")",
        type=str,
    )

    args = parser.parse_args()
    if len(args.api_key) != len(args.label):
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(OVERLAYS_PATH):
        os.makedirs(OVERLAYS_PATH)

    resources = []
    for key, label in zip(args.api_key, args.label):
        resources.append(generate(key, label, args.image_version))

    path = os.path.join(OVERLAYS_PATH, "kustomization.yaml")
    with open(path, "w+") as fp:  # pylint: disable=invalid-name
        fp.write("resources:\n")
        fp.writelines([f"  - ./{x}\n" for x in resources])

    print(
        "Successfully created following workers overlays:\n",
        "\n ".join([f"  - {x}" for x in resources]),
    )


if __name__ == "__main__":
    main()
