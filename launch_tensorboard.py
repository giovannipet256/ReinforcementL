#!/usr/bin/env python
# ============================================================
# launch_tensorboard.py — Avvia TensorBoard sui logs
# ============================================================

import subprocess
import sys

print("\n" + "="*70)
print("🚀 Avvio TensorBoard...")
print("="*70)
print("\n📊 Accedi a: http://localhost:6006")
print("\nPremi Ctrl+C per fermare TensorBoard\n")
print("="*70 + "\n")

try:
    subprocess.run(["tensorboard", "--logdir=./logs"], check=True)
except KeyboardInterrupt:
    print("\n\n❌ TensorBoard fermato.")
    sys.exit(0)
except FileNotFoundError:
    print("\n❌ Errore: TensorBoard non trovato!")
    print("Installa con: pip install tensorboard")
    sys.exit(1)
