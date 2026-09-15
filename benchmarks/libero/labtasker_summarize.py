#!/usr/bin/env python3
"""Summarize one LIBERO eval submission from Labtasker."""

import argparse
import sys
from pathlib import Path

from labtasker_runtime import summarize


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--queue")
    args = parser.parse_args(argv)
    return summarize(args.submission_id, args.output_dir.expanduser().absolute(), queue=args.queue)


if __name__ == "__main__":
    sys.exit(main())
