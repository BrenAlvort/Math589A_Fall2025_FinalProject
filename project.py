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
# 3. SVD features (your friend's stable version)
# =========================================================
def svd_features(image, p, tol=1e-12):
    """
    Feature vector length p+2:
      [ s1/sum(s), ..., sp/sum(s), r90, r95 ]
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

    e = s * s
    total = float(np.sum(e))
    if total > tol:
        c = np.cumsum(e) / total
        r90 = float(np.searchsorted(c, 0.90) + 1)
        r95 = float(np.searchsorted(c, 0.95) + 1)
    else:
        r90 = 0.0
        r95 = 0.0

    return np.concatenate([lead, np.array([r90, r95], dtype=float)])


# =========================================================
# Helper: fit LDA weights with ridge, return (w, m0, m1, mu0, mu1)
# =========================================================
def _lda_fit_core(X, y):
    """
    Assumes y in {0,1}.
    Returns w, mu0, mu1, m0, m1.
    """
    X0 = X[y == 0]
    X1 = X[y == 1]
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
    Sw = Sw + lam * np.eye(d)

    w = np.linalg.solve(Sw, (mu1 - mu0))

    m0 = float(mu0 @ w)
    m1 = float(mu1 @ w)

    # enforce m1 >= m0 for consistent ">= threshold => class 1"
    if m1 < m0:
        w = -w
        m0, m1 = -m0, -m1
        mu0, mu1 = mu0, mu1  # unchanged; projections already fixed

    return w, mu0, mu1, m0, m1


# =========================================================
# 4. LDA training (SAFE prior correction chosen by CV)
# =========================================================
def lda_train(X, y):
    """
    Train LDA and choose a safe prior-correction strength beta via deterministic CV.

    threshold(beta) = 0.5*(m0+m1) - beta * log(pi1/pi0)

    beta is chosen from a small grid; if correction hurts, beta=0 wins automatically.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.size != X.shape[0]:
        raise ValueError("y must have length N")

    # IMPORTANT: keep labels as 0/1 to match autograder.
    # If y is not 0/1, map it but return a classifier that still outputs 0/1.
    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError("lda_train expects exactly two classes")

    # Map to {0,1} deterministically (smaller value -> 0, larger -> 1)
    y01 = (y == classes.max()).astype(int)

    N = X.shape[0]
    idx = np.arange(N)
    K = 5 if N >= 80 else 3
    folds = np.array_split(idx, K)

    betas = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)

    best_beta = 0.0
    best_acc = -1.0

    for beta in betas:
        correct = 0
        total = 0
        for k in range(K):
            val_idx = folds[k]
            tr_idx = np.hstack([folds[j] for j in range(K) if j != k])

            Xtr = X[tr_idx]
            ytr = y01[tr_idx]
            Xva = X[val_idx]
            yva = y01[val_idx]

            # Must contain both classes
            if np.all(ytr == 0) or np.all(ytr == 1):
                continue

            w, mu0, mu1, m0, m1 = _lda_fit_core(Xtr, ytr)

            # midpoint
            thr = 0.5 * (m0 + m1)

            # prior correction (bounded by beta; beta=0 is baseline)
            pi0 = max(np.mean(ytr == 0), 1e-12)
            pi1 = max(np.mean(ytr == 1), 1e-12)
            thr = thr - beta * np.log(pi1 / pi0)

            pred = (Xva @ w >= thr).astype(int)
            correct += int(np.sum(pred == yva))
            total += yva.size

        if total > 0:
            acc = correct / total
            # tie-break: prefer smaller beta (more conservative)
            if acc > best_acc + 1e-12 or (abs(acc - best_acc) <= 1e-12 and beta < best_beta):
                best_acc = acc
                best_beta = beta

    # Fit on full data using best_beta
    w, mu0, mu1, m0, m1 = _lda_fit_core(X, y01)
    thr = 0.5 * (m0 + m1)

    pi0 = max(np.mean(y01 == 0), 1e-12)
    pi1 = max(np.mean(y01 == 1), 1e-12)
    thr = thr - best_beta * np.log(pi1 / pi0)

    return w, float(thr)


# =========================================================
# 5. LDA prediction
# =========================================================
def lda_predict(X, w, threshold):
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if X.shape[1] != w.size:
        raise ValueError("Dimension mismatch between X and w")
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

    p = min(20, min(X_train.shape[1], X_train.shape[2]))
    Xf_train = np.vstack([svd_features(img, p) for img in X_train])
    Xf_test  = np.vstack([svd_features(img, p) for img in X_test])

    w, thr = lda_train(Xf_train, y_train)
    y_pred = lda_predict(Xf_test, w, thr)
    print("Example accuracy:", float(np.mean(y_pred == (y_test == np.max(np.unique(y_test))).astype(int))))


if __name__ == "__main__":
    _example_run()
