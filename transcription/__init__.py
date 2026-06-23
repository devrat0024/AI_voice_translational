# Root-level transcription package shim
# Redirects imports to data_transcriptor.transcription
import sys
from pathlib import Path

# Ensure the project root (parent of this package) is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
