"""Thin CLI for regional-montage participant statistics."""

import runpy


if __name__ == "__main__":
    runpy.run_module(
        "bench.analysis.regional_montage_v2_participant_statistics",
        run_name="__main__",
    )
