import numpy as np

# =========================================================
# 1. Power method for dominant eigenpair
# =========================================================
def power_method(A: np.ndarray, x0: np.ndarray, maxit: int, tol: float):
    """
    Approximate dominant eigenvalue/eigenvector of (assumed) real symmetric A.
    Returns (lam, v, iters).
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    n = A.shape[0]

    x = np.asarray(x0, dtype=float).reshape(-1)
    if x.size != n:
        raise ValueError("x0 must have length n")
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
def svd_compress(image: np.ndarray, k: int):
    """
    Returns (image_k, rel_error, compression_ratio).
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError("image must be 2D")
    m, n = image.shape
    rmax = min(m, n)
    if not (1 <= k <= rmax):
        raise ValueError(f"k must be in [1, {rmax}]")

    U, s, Vh = np.linalg.svd(image, full_matrices=False)
    image_k = (U[:, :k] * s[:k]) @ Vh[:k, :]

    denom = np.linalg.norm(image, ord="fro")
    rel_error = 0.0 if denom == 0 else float(np.linalg.norm(image - image_k, ord="fro") / denom)
    compression_ratio = float(k * (m + n + 1) / (m * n))

    return image_k, rel_error, compression_ratio


# =========================================================
# 3. SVD-based feature extraction (corrected)
# =========================================================
def svd_features(image: np.ndarray, p: int, tol: float = 1e-12):
    """
    Feature vector:
      [ s1/sum(s), ..., sp/sum(s), r_0.9, r_0.95 ]
    where r_alpha is smallest r with sum_{i<=r} s_i^2 >= alpha * sum_i s_i^2.
    """
    A = np.asarray(image, dtype=float)
    if A.ndim != 2:
        raise ValueError("image must be 2D")
    m, n = A.shape
    rmax = min(m, n)
    if not (1 <= p <= rmax):
        raise ValueError(f"p must be in [1, {rmax}]")

    s = np.linalg.svd(A, full_matrices=False, compute_uv=False)

    s_sum = float(np.sum(s))
    top = (s[:p] / s_sum) if s_sum > tol else np.zeros(p, dtype=float)

    s2 = s * s
    total = float(np.sum(s2))
    if total > tol:
        c = np.cumsum(s2) / total
        r_09 = float(np.searchsorted(c, 0.90) + 1)
        r_095 = float(np.searchsorted(c, 0.95) + 1)
    else:
        r_09 = 0.0
        r_095 = 0.0

    return np.concatenate([top, np.array([r_09, r_095], dtype=float)])


# =========================================================
# 4. Two-class LDA: training (corrected + stabilized)
# =========================================================
def lda_train(X: np.ndarray, y: np.ndarray):
    """
    Returns (w, threshold). Predict 1 if X@w >= threshold else 0.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if y.size != X.shape[0]:
        raise ValueError("y must match number of rows in X")

    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError("lda_train expects exactly two classes")

    # Map to {0,1} deterministically
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

    # Ensure class-1 projects higher than class-0 for stable threshold rule
    if m1 < m0:
        w = -w
        threshold = -threshold

    return w, float(threshold)


# =========================================================
# 5. Two-class LDA: prediction
# =========================================================
def lda_predict(X: np.ndarray, w: np.ndarray, threshold: float):
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != w.size:
        raise ValueError("Dimension mismatch X and w")
    return (X @ w >= float(threshold)).astype(int)


# =========================================================
# Local smoke test (not used by autograder)
# =========================================================
def _example_run():
    try:
        data = np.load("project_data_example.npz")
    except OSError:
        print("No example data file 'project_data_example.npz' found.")
        return

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    p = min(20, min(X_train.shape[1], X_train.shape[2]))

    Xf_train = np.vstack([svd_features(img, p) for img in X_train])
    Xf_test = np.vstack([svd_features(img, p) for img in X_test])

    w, thr = lda_train(Xf_train, y_train)
    y_pred = lda_predict(Xf_test, w, thr)

    # Map y_test to {0,1} in the same way as training
    classes = np.unique(y_train)
    y_test01 = (y_test == classes.max()).astype(int)
    acc = float(np.mean(y_pred == y_test01))
    print(f"Example test accuracy: {acc:.3f}")


if __name__ == "__main__":
    _example_run()
