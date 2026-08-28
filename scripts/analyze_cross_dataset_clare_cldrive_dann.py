"""Thin CLI for the CL-Drive/CLARE DANN analysis."""

import runpy


if __name__ == "__main__":
    runpy.run_module(
        "bench.analysis.cross_dataset_clare_cldrive_dann_statistics",
        run_name="__main__",
    )
