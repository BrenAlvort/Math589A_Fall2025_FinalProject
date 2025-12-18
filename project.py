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
        if lam_old is not None:
            if abs(lam - lam_old) <= tol * max(1.0, abs(lam)):
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
    r = s.size
    k_eff = int(min(k, r))
    Ak = (U[:, :k_eff] * s[:k_eff]) @ Vt[:k_eff, :]

    denom = np.linalg.norm(A, ord="fro")
    rel_error = 0.0 if denom == 0 else float(np.linalg.norm(A - Ak, ord="fro") / denom)
    return Ak, rel_error


# =========================================================
# 3. Feature vector from image singular values (head+tail)
# =========================================================
def svd_features(image, p, tol=1e-12):
    """
    Feature vector (length p+2):

      first p_head entries:   head singular values normalized by Frobenius norm
      next  p_tail entries:   tail singular values normalized by Frobenius norm
      extra1: spectral entropy of energy proportions
      extra2: effective rank = exp(entropy)

    This captures BOTH global structure (head) and texture/noise (tail).
    """
    A = np.asarray(image, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("image must be a 2D array")

    m, n = A.shape
    rmax = min(m, n)
    if p < 1 or p > rmax:
        raise ValueError("p must satisfy 1 <= p <= min(m,n)")

    S = np.linalg.svd(A, full_matrices=False, compute_uv=False)

    # Frobenius energy and stability
    e = S * S
    tot = float(e.sum())
    if tot <= tol:
        return np.zeros(p + 2, dtype=np.float32)

    frob = np.sqrt(tot)

    # Split budget
    p_head = p // 2
    p_tail = p - p_head

    head = (S[:p_head] / frob) if p_head > 0 else np.empty((0,), dtype=np.float64)

    # Tail: take the smallest singular values (texture/noise regime)
    # Reverse end ensures we include the tiniest values first.
    if p_tail > 0:
        tail = (S[-p_tail:] / frob)
    else:
        tail = np.empty((0,), dtype=np.float64)

    lead = np.concatenate([head, tail]).astype(np.float32, copy=False)

    # Spectral entropy (continuous, stable)
    pi = e / tot
    eps = 1e-12
    H = -float(np.sum(pi * np.log(pi + eps)))
    eff_rank = float(np.exp(H))

    return np.concatenate([lead, np.array([H, eff_rank], dtype=np.float32)])


# =========================================================
# 4. Two-class LDA training (simple + mild shrinkage)
# =========================================================
def lda_train(X, y):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.size != X.shape[0]:
        raise ValueError("y must have length N matching X.shape[0]")

    X0 = X[y == 0]
    X1 = X[y == 1]
    if X0.shape[0] == 0 or X1.shape[0] == 0:
        raise ValueError("Both classes must have at least one sample")

    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)

    X0c = X0 - mu0
    X1c = X1 - mu1
    Sw = (X0c.T @ X0c) + (X1c.T @ X1c)

    d = Sw.shape[0]
    tr = float(np.trace(Sw))
    if tr <= 0.0:
        tr = 1.0

    # Mild shrinkage: improves generalization without being “fancy”
    gamma = 0.05
    Sw = (1.0 - gamma) * Sw + gamma * (tr / d) * np.eye(d, dtype=np.float64)

    # Tiny diagonal loading for numerical safety
    Sw = Sw + (1e-8 * tr / d) * np.eye(d, dtype=np.float64)

    w = np.linalg.solve(Sw, (mu1 - mu0))

    m0 = float(mu0 @ w)
    m1 = float(mu1 @ w)
    threshold = 0.5 * (m0 + m1)

    # Ensure class-1 scores higher for >= rule
    if m1 < m0:
        w = -w
        threshold = -threshold

    return w, float(threshold)


# =========================================================
# 5. Prediction
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
    Xf_test = np.vstack([svd_features(img, p) for img in X_test])

    w, thr = lda_train(Xf_train, y_train)
    y_pred = lda_predict(Xf_test, w, thr)
    print("Example accuracy:", float(np.mean(y_pred == y_test)))


if __name__ == "__main__":
    _example_run()
