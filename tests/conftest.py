"""Pytest configuration - project root sys.path bootstrap.

Mirrors the bootstrap pattern already used in pipelines/etl_pipeline.py,
pipelines/train_pipeline.py, pipelines/decision_pipeline.py, and api/main.py
(see PHASE3_LOG.md, sự cố #1) so `from src.xxx import yyy` works during
`pytest` collection regardless of the working directory pytest is invoked
from. Kept consistent with the rest of the project instead of introducing a
new mechanism (e.g. pyproject.toml pythonpath) that nothing else uses.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
