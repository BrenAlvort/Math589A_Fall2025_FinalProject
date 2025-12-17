import numpy as np

def svd_features(image, p, tol=1e-12):
    """
    Feature vector (length p+2):
      [ log(sigma1/sigma1+eps), ..., log(sigmap/sigma1+eps), r90, r95 ]
    """
    A = np.asarray(image, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("image must be a 2D array")
    m, n = A.shape
    rmax = min(m, n)
    if p < 1 or p > rmax:
        raise ValueError("p must satisfy 1 <= p <= min(m,n)")

    S = np.linalg.svd(A, full_matrices=False, compute_uv=False)

    # Energy ranks (Frobenius): sum sigma_i^2
    energy = S * S
    total = float(energy.sum())
    if total <= tol:
        # Degenerate image
        lead = np.zeros(p, dtype=np.float32)
        return np.concatenate([lead, np.array([0.0, 0.0], dtype=np.float32)])

    c = np.cumsum(energy) / total
    r90 = float(np.searchsorted(c, 0.90) + 1)
    r95 = float(np.searchsorted(c, 0.95) + 1)

    # Log-spectrum ratios: very robust to brightness/contrast scaling
    s1 = float(S[0])
    denom = s1 if s1 > tol else float(np.sqrt(total))
    eps = 1e-12
    ratios = S[:p] / denom
    lead = np.log(ratios + eps).astype(np.float32, copy=False)

    return np.concatenate([lead, np.array([r90, r95], dtype=np.float32)])


def lda_train(X, y):
    """
    Two-class LDA with shrinkage regularization.
    Returns w, threshold (predict 1 if X@w >= threshold else 0).
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

    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)

    X0c = X0 - mu0
    X1c = X1 - mu1
    Sw = (X0c.T @ X0c) + (X1c.T @ X1c)

    d = Sw.shape[0]
    tr = float(np.trace(Sw))
    if tr <= 0.0:
        tr = 1.0

    # Shrinkage: Sw_shrunk = (1-gamma) Sw + gamma * (tr(Sw)/d) I
    # gamma in [0,1]; 0.05–0.2 is usually a good range.
    gamma = 0.10
    iso = (tr / d) * np.eye(d, dtype=np.float64)
    Sw_shrunk = (1.0 - gamma) * Sw + gamma * iso

    # Small diagonal loading to be safe
    Sw_shrunk = Sw_shrunk + (1e-10 * tr / d) * np.eye(d, dtype=np.float64)

    w = np.linalg.solve(Sw_shrunk, (mu1 - mu0))

    m0 = float(mu0 @ w)
    m1 = float(mu1 @ w)
    threshold = 0.5 * (m0 + m1)

    # Ensure class 1 tends to have larger score
    if m1 < m0:
        w = -w
        threshold = -threshold

    return w, float(threshold)


def lda_predict(X, w, threshold):
    X = np.asarray(X, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    return (X @ w >= float(threshold)).astype(int)
