import numpy as np

# =========================================================
# 1. Power method for dominant eigenpair
# =========================================================
def power_method(A, x0, maxit, tol):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    n = A.shape[0]

    x = np.asarray(x0, dtype=float).reshape(-1)
    if x.size != n:
        raise ValueError("x0 must have shape (n,)")
    nx = np.linalg.norm(x)
    if nx == 0:
        x = np.ones(n, dtype=float)
        nx = np.linalg.norm(x)
    x /= nx

    lam_prev = float(x @ (A @ x))
    eps = np.finfo(float).eps

    iters = 0
    for iters in range(1, int(maxit) + 1):
        y = A @ x
        ny = np.linalg.norm(y)
        if ny == 0:
            return 0.0, x, iters
        x = y / ny

        lam = float(x @ (A @ x))
        rel_err = abs(lam - lam_prev) / max(abs(lam), eps)
        if rel_err < tol:
            return lam, x, iters
        lam_prev = lam

    return lam_prev, x, iters


# =========================================================
# 2. Rank-k image compression via SVD
# =========================================================
def svd_compress(image, k):
    A = np.asarray(image, dtype=float)
    if A.ndim != 2:
        raise ValueError("image must be 2D")
    m, n = A.shape
    r = min(m, n)
    k = int(k)
    if not (1 <= k <= r):
        raise ValueError(f"k must be in [1,{r}]")

    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    Ak = (U[:, :k] * s[:k]) @ Vt[:k, :]

    denom = np.linalg.norm(A, ord="fro")
    rel_err = 0.0 if denom == 0 else float(np.linalg.norm(A - Ak, ord="fro") / denom)
    comp_ratio = float(k * (m + n + 1) / (m * n))
    return Ak, rel_err, comp_ratio


# =========================================================
# 3. Local texture / "roughness" features (vectorized & correct)
# =========================================================
def block_roughness_features(image, block_size=8, tol=1e-12):
    """
    Roughness per block = mean(dx^2) + mean(dy^2),
    where dx,dy are finite differences inside each block.

    Returns: [mean, std, max] over blocks.
    """
    A = np.asarray(image, dtype=float)
    if A.ndim != 2:
        raise ValueError("image must be 2D")
    H, W = A.shape
    b = int(block_size)
    if b < 2:
        raise ValueError("block_size must be >= 2")

    # Only use full blocks to avoid shape mismatches / variable-size blocks
    Hc = (H // b) * b
    Wc = (W // b) * b
    if Hc == 0 or Wc == 0:
        return np.array([0.0, 0.0, 0.0], dtype=float)

    A = A[:Hc, :Wc]

    # Reshape into blocks: (Hc/b, b, Wc/b, b)
    blocks = A.reshape(Hc // b, b, Wc // b, b)

    # Finite differences within each block:
    dx = blocks[:, :, :, 1:] - blocks[:, :, :, :-1]   # shape (..., b-1)
    dy = blocks[:, 1:, :, :] - blocks[:, :-1, :, :]   # shape (..., b-1)

    # Mean squared diffs per block
    r = np.mean(dx * dx, axis=(1, 3)) + np.mean(dy * dy, axis=(1, 3))  # (Hc/b, Wc/b)

    r = r.reshape(-1)
    if r.size == 0:
        return np.array([0.0, 0.0, 0.0], dtype=float)

    return np.array([float(r.mean()), float(r.std()), float(r.max())], dtype=float)


# =========================================================
# 4. SVD-based feature extraction (robust, consistent)
# =========================================================
def svd_features(image, p, tol=1e-12):
    """
    Returns feature vector:
      [ s1/sum(s), ..., sp/sum(s), r95, r99, rough_mean, rough_std, rough_max ]

    - r95, r99 are computed using squared singular values (energy).
    - roughness features are computed on 8x8 blocks by default.
    """
    A = np.asarray(image, dtype=float)
    if A.ndim != 2:
        raise ValueError("image must be a 2D array")

    m, n = A.shape
    rmax = min(m, n)
    p = int(p)
    if not (1 <= p <= rmax):
        raise ValueError(f"p must be in [1,{rmax}]")

    s = np.linalg.svd(A, full_matrices=False, compute_uv=False)

    s_sum = float(np.sum(s))
    top = (s[:p] / s_sum) if s_sum > tol else np.zeros(p, dtype=float)

    e = s * s
    e_sum = float(np.sum(e))
    if e_sum > tol:
        c = np.cumsum(e) / e_sum
        r95 = float(np.searchsorted(c, 0.95) + 1)
        r99 = float(np.searchsorted(c, 0.99) + 1)
    else:
        r95 = 0.0
        r99 = 0.0

    rough = block_roughness_features(A, block_size=8, tol=tol)

    return np.concatenate([top, np.array([r95, r99], dtype=float), rough])


# =========================================================
# 5. Two-class LDA: training (stable + label-robust)
# =========================================================
def lda_train(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if y.size != X.shape[0]:
        raise ValueError("y must match number of rows in X")

    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError("lda_train expects exactly two classes")

    # Deterministic mapping to {0,1}
    y01 = (y == classes.max()).astype(int)

    X0 = X[y01 == 0]
    X1 = X[y01 == 1]
    if X0.shape[0] == 0 or X1.shape[0] == 0:
        raise ValueError("Both classes must have at least one sample")

    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)

    X0c = X0 - mu0
    X1c = X1 - mu1
    Sw = X0c.T @ X0c + X1c.T @ X1c

    d = Sw.shape[0]
    tr = float(np.trace(Sw))
    lam = 1e-6 * (tr / d if d > 0 else 1.0) + 1e-12
    Sw_reg = Sw + lam * np.eye(d)

    w = np.linalg.solve(Sw_reg, (mu1 - mu0))

    m0 = float(mu0 @ w)
    m1 = float(mu1 @ w)
    threshold = 0.5 * (m0 + m1)

    # enforce consistent direction so prediction rule is stable
    if m1 < m0:
        w = -w
        threshold = -threshold

    return w, float(threshold)


# =========================================================
# 6. Two-class LDA: prediction
# =========================================================
def lda_predict(X, w, threshold):
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != w.size:
        raise ValueError("Dimension mismatch between X and w")
    return (X @ w >= float(threshold)).astype(int)


# =========================================================
# Local smoke test / validation helper (NOT for submission)
# =========================================================
def _example_run():
    try:
        data = np.load("project_data_example.npz")
    except OSError:
        print("No example dataset found (project_data_example.npz).")
        return

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    p = min(32, min(X_train.shape[1], X_train.shape[2]))

    Xf_train = np.vstack([svd_features(img, p) for img in X_train])
    Xf_test = np.vstack([svd_features(img, p) for img in X_test])

    w, thr = lda_train(Xf_train, y_train)
    y_pred = lda_predict(Xf_test, w, thr)

    classes = np.unique(y_train)
    y_test01 = (y_test == classes.max()).astype(int)

    acc = float(np.mean(y_pred == y_test01))
    print(f"Example test accuracy: {acc:.3f}")


if __name__ == "__main__":
    # Comment out for submission if your autograder dislikes stdout.
    _example_run()
