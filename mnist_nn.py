"""Two-hidden-layer MLP for MNIST, implemented with only numpy.

Usage:
    python mnist_nn.py train        # train and save weights.npz
    python mnist_nn.py test         # load weights.npz and report test accuracy
    python mnist_nn.py train --epochs 5 --lr 0.05 --batch-size 64
"""

import argparse
import gzip
import os
import struct
import sys
import time
import urllib.request

import numpy as np

DATA_DIR = "data"
WEIGHTS_PATH = "weights.npz"
MNIST_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def download_mnist():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, fname in MNIST_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            continue
        url = f"{MNIST_BASE}/{fname}"
        print(f"Downloading {fname} ...")
        urllib.request.urlretrieve(url, path)


def load_images(path):
    with gzip.open(path, "rb") as f:
        _magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float32) / 255.0


def load_labels(path):
    with gzip.open(path, "rb") as f:
        _magic, _n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)


def load_mnist():
    download_mnist()
    X_train = load_images(os.path.join(DATA_DIR, MNIST_FILES["train_images"]))
    y_train = load_labels(os.path.join(DATA_DIR, MNIST_FILES["train_labels"]))
    X_test = load_images(os.path.join(DATA_DIR, MNIST_FILES["test_images"]))
    y_test = load_labels(os.path.join(DATA_DIR, MNIST_FILES["test_labels"]))
    return X_train, y_train, X_test, y_test


def init_params(sizes, seed=42):
    rng = np.random.default_rng(seed)
    params = {}
    for i in range(len(sizes) - 1):
        fan_in, fan_out = sizes[i], sizes[i + 1]
        params[f"W{i+1}"] = rng.normal(
            0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out)
        ).astype(np.float32)
        params[f"b{i+1}"] = np.zeros(fan_out, dtype=np.float32)
    return params


def relu(x):
    return np.maximum(0.0, x)


def relu_grad(x):
    return (x > 0).astype(np.float32)


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def forward(X, params):
    z1 = X @ params["W1"] + params["b1"]
    a1 = relu(z1)
    z2 = a1 @ params["W2"] + params["b2"]
    a2 = relu(z2)
    z3 = a2 @ params["W3"] + params["b3"]
    y_hat = softmax(z3)
    cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "y_hat": y_hat}
    return y_hat, cache


def cross_entropy(y_hat, y):
    n = y.shape[0]
    return -np.mean(np.log(y_hat[np.arange(n), y] + 1e-12))


def backward(y, cache, params):
    n = y.shape[0]
    y_oh = np.zeros_like(cache["y_hat"])
    y_oh[np.arange(n), y] = 1.0

    dz3 = (cache["y_hat"] - y_oh) / n
    dW3 = cache["a2"].T @ dz3
    db3 = dz3.sum(axis=0)

    da2 = dz3 @ params["W3"].T
    dz2 = da2 * relu_grad(cache["z2"])
    dW2 = cache["a1"].T @ dz2
    db2 = dz2.sum(axis=0)

    da1 = dz2 @ params["W2"].T
    dz1 = da1 * relu_grad(cache["z1"])
    dW1 = cache["X"].T @ dz1
    db1 = dz1.sum(axis=0)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}


def accuracy(y_hat, y):
    return float((y_hat.argmax(axis=1) == y).mean())


def train(args):
    print("Loading MNIST ...")
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    sizes = [784, args.hidden1, args.hidden2, 10]
    params = init_params(sizes, seed=args.seed)
    print(f"Architecture: {' -> '.join(str(s) for s in sizes)}")

    n = X_train.shape[0]
    rng = np.random.default_rng(args.seed)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        idx = rng.permutation(n)
        epoch_losses = []
        for start in range(0, n, args.batch_size):
            batch = idx[start:start + args.batch_size]
            X_b = X_train[batch]
            y_b = y_train[batch]

            y_hat, cache = forward(X_b, params)
            epoch_losses.append(cross_entropy(y_hat, y_b))
            grads = backward(y_b, cache, params)
            for k in params:
                params[k] -= args.lr * grads[k]

        y_test_hat, _ = forward(X_test, params)
        test_acc = accuracy(y_test_hat, y_test)
        dt = time.time() - t0
        print(
            f"epoch {epoch:2d} | loss {np.mean(epoch_losses):.4f} "
            f"| test_acc {test_acc:.4f} | {dt:.1f}s"
        )

    np.savez(WEIGHTS_PATH, **params)
    print(f"Saved weights to {WEIGHTS_PATH}")


def test(_args):
    if not os.path.exists(WEIGHTS_PATH):
        print(f"No {WEIGHTS_PATH} found — run `python mnist_nn.py train` first.")
        sys.exit(1)
    _, _, X_test, y_test = load_mnist()
    loaded = np.load(WEIGHTS_PATH)
    params = {k: loaded[k] for k in loaded.files}
    y_hat, _ = forward(X_test, params)
    print(f"Test accuracy: {accuracy(y_hat, y_test):.4f}")
    print(f"Test loss:     {cross_entropy(y_hat, y_test):.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--epochs", type=int, default=10)
    tr.add_argument("--batch-size", type=int, default=128)
    tr.add_argument("--lr", type=float, default=0.1)
    tr.add_argument("--hidden1", type=int, default=128)
    tr.add_argument("--hidden2", type=int, default=64)
    tr.add_argument("--seed", type=int, default=42)
    tr.set_defaults(func=train)

    te = sub.add_parser("test")
    te.set_defaults(func=test)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
