import numpy as np

def svd_features(image, p, tol=1e-12):
    """
    Feature vector length p+2:
      [phi_1, ..., phi_p, r90, r95]

    where:
      phi_i = log(sigma_i/sigma_1 + eps) + 0.25*(sigma_i/sigma_1)
      r_alpha is smallest r s.t. sum_{i<=r} sigma_i^2 >= alpha * sum_i sigma_i^2
    """
    A = np.asarray(image, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("image must be a 2D array")
    m, n = A.shape
    rmax = min(m, n)
    if not (1 <= p <= rmax):
        raise ValueError("p must satisfy 1 <= p <= min(m,n)")

    # singular values only, reduced
    S = np.linalg.svd(A, full_matrices=False, compute_uv=False)

    # energy ranks (Frobenius)
    energy = S * S
    total = float(energy.sum())
    if total <= tol:
        lead = np.zeros(p, dtype=np.float32)
        return np.concatenate([lead, np.array([0.0, 0.0], dtype=np.float32)])

    c = np.cumsum(energy) / total
    r90 = float(np.searchsorted(c, 0.90) + 1)
    r95 = float(np.searchsorted(c, 0.95) + 1)

    s1 = float(S[0])
    if s1 <= tol:
        s1 = float(np.sqrt(total))

    eps = 1e-12
    r = S[:p] / s1

    # robust spectrum-shape encoding
    lead = (np.log(r + eps) + 0.25 * r).astype(np.float32, copy=False)

    return np.concatenate([lead, np.array([r90, r95], dtype=np.float32)])


def lda_train(X, y):
    """
    Two-class LDA with:
      - per-feature standardization (z-score) for robustness
      - shrinkage covariance for stability
      - prior-adjusted threshold (helps if class imbalance)

    Returns w, threshold for rule: predict 1 if X@w >= threshold else 0.
    """
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

    # ---- Standardize features (z-score) on training data ----
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd[sd < 1e-12] = 1.0  # avoid division by ~0

    Xs = (X - mu) / sd
    X0s = Xs[y == 0]
    X1s = Xs[y == 1]

    mu0 = X0s.mean(axis=0)
    mu1 = X1s.mean(axis=0)

    # within-class scatter
    X0c = X0s - mu0
    X1c = X1s - mu1
    Sw = (X0c.T @ X0c) + (X1c.T @ X1c)

    d = Sw.shape[0]
    tr = float(np.trace(Sw))
    if tr <= 0.0:
        tr = 1.0

    # ---- Shrinkage (generalization boost on shifted test sets) ----
    # Sw_shrunk = (1-gamma) Sw + gamma * (tr(Sw)/d) I
    gamma = 0.12
    Sw_shrunk = (1.0 - gamma) * Sw + gamma * (tr / d) * np.eye(d, dtype=np.float64)

    # tiny diagonal loading
    Sw_shrunk += (1e-10 * tr / d) * np.eye(d, dtype=np.float64)

    w_s = np.linalg.solve(Sw_shrunk, (mu1 - mu0))

    # ---- Prior-adjusted threshold in standardized space ----
    n0 = float(X0s.shape[0])
    n1 = float(X1s.shape[0])
    pi0 = n0 / (n0 + n1)
    pi1 = n1 / (n0 + n1)

    # threshold for score z = w_s^T x_s:
    # classify 1 if z >= 0.5*w_s^T(mu0+mu1) - log(pi1/pi0)
    tau_s = 0.5 * float(w_s @ (mu0 + mu1)) - np.log(pi1 / pi0)

    # ---- Fold standardization back into (w, threshold) for raw X ----
    # w_s^T((x-mu)/sd) >= tau_s
    # (w_s/sd)^T x >= tau_s + w_s^T(mu/sd)
    w = w_s / sd
    offset = float(w_s @ (mu / sd))
    threshold = float(tau_s + offset)

    # keep orientation consistent: class 1 higher score (empirically stabilizes)
    m0 = float((X0 @ w).mean())
    m1 = float((X1 @ w).mean())
    if m1 < m0:
        w = -w
        threshold = -threshold

    return w, threshold


def lda_predict(X, w, threshold):
    X = np.asarray(X, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    return (X @ w >= float(threshold)).astype(int)
