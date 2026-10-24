"""CLI for the separately frozen V1.4-D2 diagnostic."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from a3_ultrasound.d2_diagnostic import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    main(args.config, args.output)
