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
# pylint: disable=protected-access

"""Unit test for base module"""
import importlib
import unittest
import unittest.mock
import redis

from parallel_accel.shared.redis import base


class TestBaseRedisStore(unittest.TestCase):
    """Tests BaseRedisStore class behavior."""

    @classmethod
    def setUpClass(cls) -> None:
        """See base class documentation."""
        cls.patchers = []

        cls.mocked_redis = unittest.mock.Mock(spec=redis.Redis)
        cls.mocked_redis.return_value = cls.mocked_redis
        patcher = unittest.mock.patch("redis.Redis", cls.mocked_redis)
        cls.patchers.append(patcher)

        for patcher in cls.patchers:
            patcher.start()

        importlib.reload(base)

    @classmethod
    def tearDownClass(cls) -> None:
        """See base class documentation."""
        for patcher in cls.patchers:
            patcher.stop()

    def tearDown(self) -> None:
        """See base class documentation."""
        for mock in [x for x in dir(self) if x.startswith("mocked_")]:
            getattr(self, mock).reset_mock()

    def test_init(self) -> None:
        """Tests __init__ method."""
        instances = list(base.RedisInstances)
        host = "localhost"
        port = 6379

        # Run test
        obj = base.BaseRedisStore(instances, host, port)

        # Verifciation
        self.assertTrue(isinstance(obj, base.BaseRedisStore))
        expected_call_args = [
            (
                (),
                {
                    "db": x.value,
                    "decode_responses": True,
                    "host": host,
                    "port": port,
                },
            )
            for x in instances
        ]
        self.assertEqual(self.mocked_redis.call_args_list, expected_call_args)
