"""
The long benchmark of ``benchmarks/long_benchmark.py`` as a test: the results must be those of the
reference. It takes a few minutes, so it runs only on request::

    TSLIES_LONG_BENCHMARK=1 pytest tests/test_long_benchmark.py
"""

import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("TSLIES_LONG_BENCHMARK"),
                                reason="long benchmark: set TSLIES_LONG_BENCHMARK=1 to run it")


def test_results_are_those_of_the_reference():
    from benchmarks.long_benchmark import compare, load_reference, run

    reference = load_reference()
    if reference is None:
        pytest.skip("no reference yet: run python benchmarks/long_benchmark.py, then --accept")
    lines, same = compare(run(verbose=False), reference)
    assert same, "\n".join(["different from the reference:", *lines])
