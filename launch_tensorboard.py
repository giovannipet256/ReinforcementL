#!/usr/bin/env python
# ============================================================
# launch_tensorboard.py - Avvia TensorBoard sui logs
# ============================================================

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
LOGS_DIR = PROJECT_ROOT / "logs"

print("\n" + "=" * 70)
print("Avvio TensorBoard...")
print("=" * 70)
print("\nAccedi a: http://localhost:6006")
print("\nPremi Ctrl+C per fermare TensorBoard\n")
print("=" * 70 + "\n")

LOGS_DIR.mkdir(exist_ok=True)

try:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tensorboard.main",
            f"--logdir={LOGS_DIR}",
            "--port=6006",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
except KeyboardInterrupt:
    print("\n\nTensorBoard fermato.")
    sys.exit(0)
except ModuleNotFoundError:
    print("\nErrore: TensorBoard non trovato nell'environment Python attivo.")
    print("Installa con: python -m pip install tensorboard")
    sys.exit(1)
except subprocess.CalledProcessError as exc:
    sys.exit(exc.returncode)
