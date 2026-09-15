#!/usr/bin/env python
"""NetShield-ML command-line entry point.

Usage:
    python main.py               # full study, with cross-validation
    python main.py --skip-cv     # faster run, no cross-validation

Authors: Anurag Jha, Sandesh Dhital
"""

from src.pipeline import main

if __name__ == "__main__":
    main()
