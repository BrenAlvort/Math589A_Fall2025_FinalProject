import numpy as np

# =========================================================
# 1. Power method for dominant eigenpair (symmetric matrix)
# =========================================================
def power_method(A, x0, maxit, tol):
    """
    Approximate the dominant eigenvalue/eigenvector of a real symmetric matrix A
    using the power method.

    Parameters
    ----------
    A : (n, n) ndarray
        Real symmetric matrix.
    x0 : (n,) ndarray
        Initial guess (must be nonzero).
    maxit : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on successive eigenvalue estimates.

    Returns
    -------
    lam : float
        Approximate dominant eigenvalue.
    x : (n,) ndarray
        Approximate dominant eigenvector (unit 2-norm).
    """
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
    """
    Compute a rank-k approximation of a grayscale image using SVD.

    Parameters
    ----------
    image : (m, n) ndarray
        Image matrix.
    k : int
        Target rank (k >= 1).

    Returns
    -------
    image_k : (m, n) ndarray
        Rank-k approximation.
    rel_error : float
        Relative Frobenius error ||A - A_k||_F / ||A||_F.
    """
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
# 3. Build feature vector from image singular values
#    (Leaderboard-stable version)
# =========================================================
def svd_features(image, p, tol=1e-12):
    """
    Feature vector (length p+2):
        [ sigma_1/||A||_F, ..., sigma_p/||A||_F, r90, r95 ]

    where r_alpha is the smallest r such that:
        sum_{i=1}^r sigma_i^2 >= alpha * sum_i sigma_i^2
    """
    A = np.asarray(image, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("image must be a 2D array")

    m, n = A.shape
    rmax = min(m, n)
    if p < 1 or p > rmax:
        raise ValueError("p must satisfy 1 <= p <= min(m,n)")

    # singular values only; reduced factorization
    S = np.linalg.svd(A, full_matrices=False, compute_uv=False)

    energy = S * S
    total_energy = float(energy.sum())
    if total_energy <= tol:
        # Degenerate image
        return np.zeros(p + 2, dtype=np.float32)

    # Effective ranks
    c = np.cumsum(energy) / total_energy
    r90 = float(np.searchsorted(c, 0.90) + 1)
    r95 = float(np.searchsorted(c, 0.95) + 1)

    # Frobenius normalization (stable under global intensity scaling)
    frob = np.sqrt(total_energy)
    lead = (S[:p] / frob).astype(np.float32, copy=False)

    return np.concatenate([lead, np.array([r90, r95], dtype=np.float32)])


# =========================================================
# 4. Two-class LDA: training
# =========================================================
def lda_train(X, y):
    """
    Train a two-class Linear Discriminant Analysis classifier.

    Parameters
    ----------
    X : (N, d) ndarray
        Feature matrix.
    y : (N,) ndarray
        Binary labels (0/1).

    Returns
    -------
    w : (d,) ndarray
        Discriminant direction.
    threshold : float
        Predict 1 if (X @ w) >= threshold else 0.
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

    # Within-class scatter
    X0c = X0 - mu0
    X1c = X1 - mu1
    Sw = (X0c.T @ X0c) + (X1c.T @ X1c)

    # Light Tikhonov regularization (robust but not aggressive)
    d = Sw.shape[0]
    tr = float(np.trace(Sw))
    lam = (1e-6 * tr / d) if tr > 0.0 else 1e-6
    Sw = Sw + lam * np.eye(d, dtype=np.float64)

    # Solve Sw w = (mu1 - mu0)
    w = np.linalg.solve(Sw, (mu1 - mu0))

    # Midpoint threshold in projected space
    m0 = float(mu0 @ w)
    m1 = float(mu1 @ w)
    threshold = 0.5 * (m0 + m1)

    # Ensure class-1 scores higher
    if m1 < m0:
        w = -w
        threshold = -threshold

    return w, float(threshold)


# =========================================================
# 5. Two-class LDA: prediction
# =========================================================
def lda_predict(X, w, threshold):
    """
    Predict labels for samples using the trained LDA model.

    Parameters
    ----------
    X : (N, d) ndarray
        Feature matrix.
    w : (d,) ndarray
        Discriminant direction from lda_train.
    threshold : float
        Threshold from lda_train.

    Returns
    -------
    y_pred : (N,) ndarray of int
        Predicted labels in {0,1}.
    """
    X = np.asarray(X, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
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
    """
    Run a tiny end-to-end test if 'project_data_example.npz' exists.
    This is for local testing only and will NOT be called by the autograder.
    """
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
