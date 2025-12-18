import numpy as np

def power_method(A, x0, maxit, tol):
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    n = A.shape[0]

    x = np.asarray(x0, dtype=np.float64).reshape(-1)
    if x.size != n:
        raise ValueError("x0 must have shape (n,)")
    if np.linalg.norm(x) == 0:
        x = np.ones(n, dtype=np.float64)
    x /= np.linalg.norm(x)

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


def svd_compress(image, k):
    A = np.asarray(image, dtype=np.float64)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    k = min(int(k), s.size)
    Ak = (U[:, :k] * s[:k]) @ Vt[:k, :]
    denom = np.linalg.norm(A, "fro")
    rel_error = 0.0 if denom == 0 else float(np.linalg.norm(A - Ak, "fro") / denom)
    return Ak, rel_error


def svd_features(image, p, tol=1e-12):
    A = np.asarray(image, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("image must be 2D")
    m, n = A.shape
    if not (1 <= p <= min(m, n)):
        raise ValueError("invalid p")

    S = np.linalg.svd(A, full_matrices=False, compute_uv=False)

    e = S * S
    tot = float(e.sum())
    if tot <= tol:
        return np.zeros(p + 2, dtype=np.float32)

    # ranks from full energy (stable)
    cum_full = np.cumsum(e) / tot
    r95 = float(np.searchsorted(cum_full, 0.95) + 1)
    r99 = float(np.searchsorted(cum_full, 0.99) + 1)

    # head energy distribution (more discriminative, less noisy than raw sigmas)
    eh = e[:p]
    denom = float(eh.sum())
    if denom <= tol:
        lead = np.zeros(p, dtype=np.float32)
    else:
        lead = (eh / denom).astype(np.float32, copy=False)

    return np.concatenate([lead, np.array([r95, r99], dtype=np.float32)])


def lda_train(X, y):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).reshape(-1)

    X0 = X[y == 0]
    X1 = X[y == 1]
    if X0.size == 0 or X1.size == 0:
        raise ValueError("Both classes must be present")

    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)

    X0c = X0 - mu0
    X1c = X1 - mu1
    Sw = X0c.T @ X0c + X1c.T @ X1c

    d = Sw.shape[0]
    tr = float(np.trace(Sw))
    lam = (1e-6 * tr / d) if tr > 0 else 1e-6
    Sw = Sw + lam * np.eye(d)

    w = np.linalg.solve(Sw, mu1 - mu0)
    m0 = float(mu0 @ w)
    m1 = float(mu1 @ w)
    thr = 0.5 * (m0 + m1)
    if m1 < m0:
        w = -w
        thr = -thr
    return w, float(thr)


def lda_predict(X, w, threshold):
    X = np.asarray(X, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    return (X @ w >= float(threshold)).astype(int)
