"""Summarize one RoboTwin eval submission from Labtasker."""

import argparse
from pathlib import Path

import labtasker_runtime as rt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--queue")
    args = parser.parse_args(argv)
    return rt.summarize(args.submission_id, args.output_dir.expanduser().absolute(), queue=args.queue)


if __name__ == "__main__":
    raise SystemExit(main())
