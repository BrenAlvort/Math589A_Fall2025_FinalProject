import numpy as np

# =========================================================
# 1. Power method for dominant eigenpair (symmetric matrix)
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
    x = x / nx

    lam_old = None
    lam = float(x @ (A @ x))

    for _ in range(int(maxit)):
        z = A @ x
        nz = np.linalg.norm(z)
        if nz == 0:
            return 0.0, x
        x = z / nz

        lam = float(x @ (A @ x))
        if lam_old is not None:
            if abs(lam - lam_old) <= tol * max(1.0, abs(lam)):
                break
        lam_old = lam

    return lam, x


# =========================================================
# 2. Rank-k image approximation using SVD
# =========================================================
def svd_compress(image, k):
    A = np.asarray(image, dtype=float)
    if A.ndim != 2:
        raise ValueError("image must be a 2D array")
    if k < 1:
        raise ValueError("k must be >= 1")

    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    r = s.size
    k_eff = int(min(k, r))
    Ak = (U[:, :k_eff] * s[:k_eff]) @ Vt[:k_eff, :]

    denom = np.linalg.norm(A, ord="fro")
    rel_error = 0.0 if denom == 0 else float(np.linalg.norm(A - Ak, ord="fro") / denom)
    return Ak, rel_error


# =========================================================
# 3. SVD-based features (keep close to your friend's)
# =========================================================
def svd_features(image, p, tol=1e-12):
    """
    Feature vector:
        [ normalized sigma_1, ..., normalized sigma_p, r90, r95 ]

    normalized sigma_i := sigma_i / sum_j sigma_j  (as in your friend's code)
    r_alpha from cumulative squared singular values.
    """
    A = np.asarray(image, dtype=float)
    if A.ndim != 2:
        raise ValueError("image must be a 2D array")
    m, n = A.shape
    rmax = min(m, n)
    if p < 1 or p > rmax:
        raise ValueError("p must satisfy 1 <= p <= min(m,n)")

    s = np.linalg.svd(A, full_matrices=False, compute_uv=False)

    s_sum = float(np.sum(s))
    lead = (s[:p] / s_sum) if s_sum > tol else np.zeros(p, dtype=float)

    s2 = s * s
    total_energy = float(np.sum(s2))
    if total_energy > tol:
        c = np.cumsum(s2) / total_energy
        r90 = float(np.searchsorted(c, 0.90) + 1)
        r95 = float(np.searchsorted(c, 0.95) + 1)
    else:
        r90 = 0.0
        r95 = 0.0

    return np.concatenate([lead, np.array([r90, r95], dtype=float)])


# =========================================================
# 4. Two-class LDA training (DIAGONAL covariance version)
# =========================================================
def lda_train(X, y):
    """
    Diagonal LDA (naive Bayes LDA):
      Sw ≈ diag(var0 + var1)
      w = (mu1 - mu0) / (var0 + var1 + ridge)

    This is often more robust on shifted test distributions than full-covariance LDA.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.size != X.shape[0]:
        raise ValueError("y must have length N")

    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError("lda_train expects exactly two classes")
    y01 = (y == classes[1]).astype(int)

    X0 = X[y01 == 0]
    X1 = X[y01 == 1]
    if X0.shape[0] == 0 or X1.shape[0] == 0:
        raise ValueError("Both classes must have at least one sample")

    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)

    # diagonal within-class scatter = per-feature variances * counts (up to a constant)
    # Use population variance for stability
    v0 = X0.var(axis=0, ddof=0)
    v1 = X1.var(axis=0, ddof=0)

    var = v0 + v1

    # ridge scaled by typical variance level
    scale = float(np.mean(var)) if np.isfinite(var).all() and float(np.mean(var)) > 0 else 1.0
    ridge = 1e-2 * scale + 1e-12  # stronger than "1e-6 trace/d" on purpose

    w = (mu1 - mu0) / (var + ridge)

    m0 = float(mu0 @ w)
    m1 = float(mu1 @ w)
    threshold = 0.5 * (m0 + m1)

    if m1 < m0:
        w = -w
        threshold = -threshold

    return w, float(threshold)


# =========================================================
# 5. Two-class LDA prediction
# =========================================================
def lda_predict(X, w, threshold):
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if X.shape[1] != w.size:
        raise ValueError("Dimension mismatch between X and w")

    z = X @ w
    return (z >= float(threshold)).astype(int)


# =========================================================
# Local smoke test (not used by autograder)
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

    p = min(20, min(X_train.shape[1], X_train.shape[2]))
    Xf_train = np.vstack([svd_features(img, p) for img in X_train])
    Xf_test  = np.vstack([svd_features(img, p) for img in X_test])

    w, thr = lda_train(Xf_train, y_train)
    y_pred = lda_predict(Xf_test, w, thr)

    acc = np.mean(y_pred == y_test)
    print(f"Example test accuracy: {acc:.3f}")


if __name__ == "__main__":
    _example_run()
