"""
Edge-Art Interactive Silhouette Mural - entry point.

Usage:
    python main.py             # auto-detect (sim on Windows, real on board)
    python main.py --no-board  # force simulation (don't launch gst-launch)
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-board", action="store_true",
                        help="Skip launching the on-board GStreamer pipeline")
    args = parser.parse_args()

    from src.agent  import Agent
    from src.ui_app import run_app

    agent = Agent()
    if not args.no_board:
        agent.start_pipeline()
    return run_app(agent)


if __name__ == "__main__":
    sys.exit(main())
