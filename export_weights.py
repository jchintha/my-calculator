"""Export weights.npz to model.js for the browser visualization.

Run after training:
    python export_weights.py
"""

import json

import numpy as np


def main():
    w = np.load("weights.npz")
    model = {}
    for k in w.files:
        arr = np.round(w[k].astype(np.float32), 5)
        model[k] = {"shape": list(arr.shape), "data": arr.flatten().tolist()}

    with open("model.js", "w") as f:
        f.write("window.MODEL = ")
        json.dump(model, f, separators=(",", ":"))
        f.write(";\n")

    total = sum(np.prod(w[k].shape) for k in w.files)
    print(f"Wrote model.js — {total:,} parameters")


if __name__ == "__main__":
    main()
