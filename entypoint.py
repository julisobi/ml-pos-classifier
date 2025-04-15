"""Entrypoint file.

This script provides the entrypoint for the POS Classifier project.
"""

import sys
import os

mode = os.getenv("MODE", "serve")

if mode == "serve":
    os.system("poetry run uvicorn app.pos_api:app --host 0.0.0.0 --port 8000")
elif mode == "train":
    os.system("poetry run python src/pos_classifier/train.py")
else:
    print("Unknown mode. Set MODE=serve or MODE=train")
    sys.exit(1)
