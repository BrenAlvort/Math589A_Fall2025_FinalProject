import numpy as np

def svd_features(image, p, tol=1e-12):
    """
    Build an SVD-based feature vector for a grayscale image.

    Output format (length p+2):
      [ s1_hat, ..., sp_hat, r95, r99 ]

    where:
      - s_i_hat are scale-invariant leading singular values (normalized by s1),
      - r95/r99 are the smallest ranks capturing 95%/99% of Frobenius energy.
    """
    A = np.asarray(image, dtype=np.float64)

    # Only singular values are needed.
    S = np.linalg.svd(A, full_matrices=False, compute_uv=False)

    # Effective ranks based on Frobenius energy: ||A||_F^2 = sum_i sigma_i^2
    energy = S * S
    total_energy = float(energy.sum())
    if total_energy <= tol:
        top = S[:p].astype(np.float32, copy=False)
        return np.concatenate([top, np.array([0.0, 0.0], dtype=np.float32)])

    cum_energy = np.cumsum(energy) / total_energy
    r95 = float(np.searchsorted(cum_energy, 0.95) + 1)
    r99 = float(np.searchsorted(cum_energy, 0.99) + 1)

    # Scale-invariant singular-value profile (helps when overall brightness/contrast changes)
    s1 = float(S[0])
    scale = s1 if s1 > tol else np.sqrt(total_energy)
    top = (S[:p] / scale).astype(np.float32, copy=False)

    return np.concatenate([top, np.array([r95, r99], dtype=np.float32)])


def lda_train(X, y):
    """
    Train a two-class LDA model.

    Returns:
      w, tau
    with prediction rule: class 1 if (w^T x) >= tau else class 0.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (N, D).")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y must be a 1D array of length N matching X.shape[0].")

    X0 = X[y == 0]
    X1 = X[y == 1]
    if X0.shape[0] == 0 or X1.shape[0] == 0:
        raise ValueError("Both classes (0 and 1) must be present in the training data.")

    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)

    # Within-class scatter matrix (pooled, unnormalized covariance numerator)
    X0c = X0 - mu0
    X1c = X1 - mu1
    Sw = (X0c.T @ X0c) + (X1c.T @ X1c)

    # Diagonal loading for numerical stability (avoids explicit inverse)
    d = Sw.shape[0]
    tr = float(np.trace(Sw))
    lam = (1e-6 * tr / d) if tr > 0.0 else 1e-6
    Sw_reg = Sw + lam * np.eye(d, dtype=np.float64)

    # LDA direction: w ∝ Sw^{-1}(mu1 - mu0)
    w = np.linalg.solve(Sw_reg, (mu1 - mu0))

    # Midpoint threshold in projected space
    tau = 0.5 * (float(w @ mu0) + float(w @ mu1))

    return w, tau


def lda_predict(X, w, threshold):
    return (X @ w >= threshold).astype(int)


def _example_run():
    """Run a tiny end-to-end test on the example dataset, if available.

    This function is for local testing only and will NOT be called by the autograder.
    """
    try:
        data = np.load("project_data.npz")
    except OSError:
        print("No example data file 'project_data.npz' found.")
        return

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)

    p = min(32, min(X_train.shape[1], X_train.shape[2]))
    print(f"Using p = {p} leading singular values for features.")

    # Build feature matrices
    def build_features(X):
        feats = []
        for img in X:
            feats.append(svd_features(img, p))
        return np.vstack(feats)

    try:
        Xf_train = build_features(X_train)
        Xf_test = build_features(X_test)
    except NotImplementedError:
        print("Implement 'svd_features' first to run this example.")
        return

    try:
        w, threshold = lda_train(Xf_train, y_train)
    except NotImplementedError:
        print("Implement 'lda_train' first to run this example.")
        return

    try:
        y_pred = lda_predict(Xf_test, w, threshold)
    except NotImplementedError:
        print("Implement 'lda_predict' first to run this example.")
        return

    accuracy = np.mean(y_pred == y_test)
    print(f"Example test accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    # This allows students to run a quick local smoke test.
    _example_run()
