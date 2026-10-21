"""Tests use the included loader, never another workspace's editable checkout."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / 'third_party/EmbodiedScore-envs'))
