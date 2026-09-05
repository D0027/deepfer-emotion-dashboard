"""
DeepFER — one-shot entry point.
Regenerates every chart/result from the notebook using your already-trained
models. No training happens here.

Usage:
    python run_evaluation.py
"""
from src.evaluate import main

if __name__ == "__main__":
    main()
