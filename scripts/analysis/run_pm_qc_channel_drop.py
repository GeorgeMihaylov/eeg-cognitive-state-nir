"""Thin F8/F7 protocol entry point using the existing resumable channel-drop runner."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.experiments.pm_f3_f4_channel_drop import main
from bench.experiments.pm_qc_channel_drop import CONFIG_PATH

if __name__ == "__main__":
    raise SystemExit(main(config_path=CONFIG_PATH))
