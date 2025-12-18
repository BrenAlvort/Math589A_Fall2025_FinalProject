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
        raise ValueError("x0 must have length n")
    nx = np.linalg.norm(x)
    if nx == 0:
        x = np.ones(n, dtype=float)
        nx = np.linalg.norm(x)
    x /= nx

    lam_prev = float(x @ (A @ x))
    eps = np.finfo(float).eps

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
    if not (1 <= int(k) <= r):
        raise ValueError(f"k must be in [1,{r}]")

    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    Ak = (U[:, :k] * s[:k]) @ Vt[:k, :]

    denom = np.linalg.norm(A, ord="fro")
    rel_err = 0.0 if denom == 0 else float(np.linalg.norm(A - Ak, ord="fro") / denom)
    comp_ratio = float(k * (m + n + 1) / (m * n))
    return Ak, rel_err, comp_ratio


# =========================================================
# 3. SVD-based features (moderate, stable choice)
# =========================================================
def svd_features(image, p, tol=1e-12):
    """
    Features:
      - cumulative proportions of singular values (length p)
      - r_0.9, r_0.95 based on squared singular values
    """
    A = np.asarray(image, dtype=float)
    if A.ndim != 2:
        raise ValueError("image must be 2D")

    s = np.linalg.svd(A, compute_uv=False)
    r = s.size
    if not (1 <= int(p) <= r):
        raise ValueError(f"p must be in [1,{r}]")
    p = int(p)

    s_sum = float(np.sum(s))
    if s_sum > tol:
        cum = np.cumsum(s) / s_sum
    else:
        cum = np.zeros_like(s, dtype=float)

    e = s * s
    e_sum = float(np.sum(e))
    if e_sum > tol:
        ecum = np.cumsum(e) / e_sum
        r90 = float(np.searchsorted(ecum, 0.90) + 1)
        r95 = float(np.searchsorted(ecum, 0.95) + 1)
    else:
        r90 = 0.0
        r95 = 0.0

    return np.hstack((cum[:p], [r90, r95])).astype(float)


# =========================================================
# 4. Two-class LDA (Bayes-consistent intercept)
# =========================================================
def lda_train(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if y.size != X.shape[0]:
        raise ValueError("y must match X rows")

    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError("lda_train expects exactly two classes")
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
    Sw += lam * np.eye(d)

    w = np.linalg.solve(Sw, (mu1 - mu0))

    # Bayes LDA intercept with empirical priors
    n0, n1 = X0.shape[0], X1.shape[0]
    pi0 = n0 / (n0 + n1)
    pi1 = n1 / (n0 + n1)
    b = -0.5 * float((mu1 + mu0) @ w) + np.log(max(pi1, 1e-15) / max(pi0, 1e-15))

    # Ensure correct orientation
    if (mu1 @ w + b) < (mu0 @ w + b):
        w = -w
        b = -b

    return w, float(b)


# =========================================================
# 5. Two-class LDA prediction
# =========================================================
def lda_predict(X, w, b):
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != w.size:
        raise ValueError("dimension mismatch X and w")
    return (X @ w + float(b) >= 0.0).astype(int)


# =========================================================
# Local smoke test (NOT used by autograder)
# =========================================================
def _example_run():
    try:
        data = np.load("project_data_example.npz")
    except OSError:
        print("No example dataset found.")
        return

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    p = min(20, min(X_train.shape[1], X_train.shape[2]))
    Xf_train = np.vstack([svd_features(img, p) for img in X_train])
    Xf_test = np.vstack([svd_features(img, p) for img in X_test])

    w, b = lda_train(Xf_train, y_train)
    y_pred = lda_predict(Xf_test, w, b)

    classes = np.unique(y_train)
    y_test01 = (y_test == classes.max()).astype(int)
    acc = np.mean(y_pred == y_test01)
    print(f"Example accuracy: {acc:.3f}")


if __name__ == "__main__":
    _example_run()
