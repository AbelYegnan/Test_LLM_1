# test_dry_run.py

import subprocess
import sys

if __name__ == "__main__":
    print("Lancement du dry-run sur tous les documents simulés...\n")
    result = subprocess.run(
        [sys.executable, "main.py", "--dry-run", "--n-runs", "50"],
        capture_output=False,
    )
    sys.exit(result.returncode)