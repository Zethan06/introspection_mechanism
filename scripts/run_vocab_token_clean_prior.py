#!/usr/bin/env python3
"""CLI entry point for model-specific clean token priors."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from introspection_core.vocab_token_clean_prior import main


if __name__ == "__main__":
    main()
