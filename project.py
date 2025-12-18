import numpy as np

# =========================================================
# 1. Power method for dominant eigenpair (symmetric matrix)
# =========================================================
def power_method(A, x0, maxit, tol):
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    n = A.shape[0]

    x = np.asarray(x0, dtype=np.float64).reshape(-1)
    if x.size != n:
        raise ValueError("x0 must have shape (n,)")

    nx = np.linalg.norm(x)
    if nx == 0:
        x = np.ones(n, dtype=np.float64)
        nx = np.linalg.norm(x)
    x /= nx

    lam_old = None
    lam = float(x @ (A @ x))
    for _ in range(int(maxit)):
        z = A @ x
        nz = np.linalg.norm(z)
        if nz == 0:
            return 0.0, x
        x = z / nz
        lam = float(x @ (A @ x))
        if lam_old is not None and abs(lam - lam_old) <= tol * max(1.0, abs(lam)):
            break
        lam_old = lam
    return lam, x


# =========================================================
# 2. Rank-k image approximation using SVD
# =========================================================
def svd_compress(image, k):
    A = np.asarray(image, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("image must be a 2D array")
    if k < 1:
        raise ValueError("k must be >= 1")

    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    k_eff = min(int(k), s.size)
    Ak = (U[:, :k_eff] * s[:k_eff]) @ Vt[:k_eff, :]

    denom = np.linalg.norm(A, ord="fro")
    rel_error = 0.0 if denom == 0 else float(np.linalg.norm(A - Ak, ord="fro") / denom)
    return Ak, rel_error


# =========================================================
# 3. SVD features (stable baseline)
# =========================================================
def svd_features(image, p, tol=1e-12):
    """
    Feature vector (length p+2):
      [ sigma_1/||A||_F, ..., sigma_p/||A||_F, r95, r99 ]
    where r_alpha uses Frobenius energy sum sigma_i^2.
    """
    A = np.asarray(image, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("image must be 2D")

    m, n = A.shape
    if not (1 <= p <= min(m, n)):
        raise ValueError("p must satisfy 1 <= p <= min(m,n)")

    S = np.linalg.svd(A, full_matrices=False, compute_uv=False)
    e = S * S
    tot = float(e.sum())
    if tot <= tol:
        return np.zeros(p + 2, dtype=np.float32)

    cum = np.cumsum(e) / tot
    r95 = float(np.searchsorted(cum, 0.95) + 1)
    r99 = float(np.searchsorted(cum, 0.99) + 1)

    frob = np.sqrt(tot)
    lead = (S[:p] / frob).astype(np.float32, copy=False)

    return np.concatenate([lead, np.array([r95, r99], dtype=np.float32)])


# =========================================================
# Helper: train LDA with blended covariance
# =========================================================
def _lda_fit_blend(X, y01, alpha, eps_scale=1e-8):
    """
    alpha in [0,1]:
      alpha=0 => full covariance
      alpha=1 => diagonal covariance
    """
    X0 = X[y01 == 0]
    X1 = X[y01 == 1]

    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)

    X0c = X0 - mu0
    X1c = X1 - mu1
    Sw_full = (X0c.T @ X0c) + (X1c.T @ X1c)

    d = Sw_full.shape[0]
    tr = float(np.trace(Sw_full))
    if tr <= 0.0:
        tr = 1.0

    # Diagonal approximation
    Sw_diag = np.diag(np.diag(Sw_full))

    Sw = (1.0 - alpha) * Sw_full + alpha * Sw_diag

    # Minimal diagonal loading
    Sw = Sw + (eps_scale * tr / d) * np.eye(d, dtype=np.float64)

    w = np.linalg.solve(Sw, (mu1 - mu0))

    m0 = float(mu0 @ w)
    m1 = float(mu1 @ w)
    thr = 0.5 * (m0 + m1)
    if m1 < m0:
        w = -w
        thr = -thr
    return w, thr


# =========================================================
# 4. LDA train with deterministic CV tuning (the “upgrade”)
# =========================================================
def lda_train(X, y):
    """
    Two-class LDA with internal deterministic CV to select alpha (blend between full and diagonal covariance).

    Returns w, threshold for prediction rule: 1 if X@w >= threshold else 0.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if y.size != X.shape[0]:
        raise ValueError("y length mismatch")

    # Expect labels {0,1}; if not, map deterministically
    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError("lda_train expects exactly two classes")
    y01 = (y == classes[1]).astype(int)

    N = X.shape[0]

    # Deterministic folds (no randomness)
    K = 5 if N >= 50 else 3
    idx = np.arange(N)
    folds = np.array_split(idx, K)

    # Small grid: low-risk, fast
    alphas = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)

    best_alpha = 0.0
    best_score = -1.0

    for a in alphas:
        correct = 0
        total = 0
        for k in range(K):
            val_idx = folds[k]
            train_idx = np.hstack([folds[j] for j in range(K) if j != k])

            Xtr = X[train_idx]
            ytr = y01[train_idx]
            Xva = X[val_idx]
            yva = y01[val_idx]

            # Must have both classes in training fold
            if np.all(ytr == 0) or np.all(ytr == 1):
                continue

            w, thr = _lda_fit_blend(Xtr, ytr, a)
            pred = (Xva @ w >= thr).astype(int)
            correct += int(np.sum(pred == yva))
            total += yva.size

        if total > 0:
            score = correct / total
            # tie-breaker: prefer more diagonal (often more robust)
            if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and a > best_alpha):
                best_score = score
                best_alpha = a

    # Fit on full data with chosen alpha
    w, thr = _lda_fit_blend(X, y01, best_alpha)
    return w, float(thr)


# =========================================================
# 5. LDA predict
# =========================================================
def lda_predict(X, w, threshold):
    X = np.asarray(X, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    return (X @ w >= float(threshold)).astype(int)


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

    p = min(32, min(X_train.shape[1], X_train.shape[2]))
    Xf_train = np.vstack([svd_features(img, p) for img in X_train])
    Xf_test  = np.vstack([svd_features(img, p) for img in X_test])

    w, thr = lda_train(Xf_train, y_train)
    y_pred = lda_predict(Xf_test, w, thr)
    print("Example accuracy:", float(np.mean(y_pred == y_test)))


if __name__ == "__main__":
    _example_run()
