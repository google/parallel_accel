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
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="remotelinear_algebra-simulator",
    version="0.1.1",
    author="ParallelAccel Team",
    description="Simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        "cloud-asic-client>=0.10",
        "linear_algebra==0.11.0",
        "jax==0.2.13",
        "jaxlib==0.1.65",
        "numpy>=1.19.4",
        "sympy==1.8",
        "graph_helper_tool>=0.4.4",
        "opt_einsum>=3.3.0"
    ],
    extras_require={"test": ["fakeredis>=1.4.4"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
