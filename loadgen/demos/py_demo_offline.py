# Copyright 2019 The MLPerf Authors. All Rights Reserved.
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

"""Python demo showing how to use the MLPerf Inference load generator bindings.
"""

from __future__ import print_function

import threading
import time

from absl import app
import mlperf_loadgen


def load_samples_to_ram(query_samples):
    del query_samples
    return


def unload_samples_from_ram(query_samples):
    del query_samples
    return



def issue_query(query_samples):
    print(f"len(query_samples): {len(query_samples)}")
    responses = []
    samples_to_complete = query_samples
    for s in samples_to_complete:
        responses.append(mlperf_loadgen.QuerySampleResponse(s.id, 0, 0))
    mlperf_loadgen.QuerySamplesComplete(responses)


def flush_queries():
    pass


def main(argv):
    del argv
    settings = mlperf_loadgen.TestSettings()
    settings.FromConfig("user.conf", "demo", "Offline")
    settings.scenario = mlperf_loadgen.TestScenario.Offline
    settings.mode = mlperf_loadgen.TestMode.PerformanceOnly
    settings.offline_expected_qps = 84000
    print("HERE")

    sut = mlperf_loadgen.ConstructSUT(issue_query, flush_queries)
    qsl = mlperf_loadgen.ConstructQSL(
        1024, 128, load_samples_to_ram, unload_samples_from_ram
    )
    print("HERE")
    mlperf_loadgen.StartTest(sut, qsl, settings)
    mlperf_loadgen.DestroyQSL(qsl)
    mlperf_loadgen.DestroySUT(sut)


if __name__ == "__main__":
    app.run(main)
