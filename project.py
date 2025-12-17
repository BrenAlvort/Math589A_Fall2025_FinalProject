import numpy as np

def svd_features(image, p, tol=1e-12):
    """
    Compute an SVD-based feature vector for a grayscale image.

    Returns:
      [ s1_hat, ..., sp_hat, r95, r99 ]
    where:
      - s_i_hat are normalized leading singular values
      - r95 is the smallest rank capturing >= 95% Frobenius energy
      - r99 is the smallest rank capturing >= 99% Frobenius energy
    """
    A = np.asarray(image, dtype=np.float64)

    # Singular values only; compute_uv=False is efficient.
    # full_matrices is irrelevant when compute_uv=False, but keep it False for clarity.
    S = np.linalg.svd(A, full_matrices=False, compute_uv=False)

    # Energy = sum sigma_i^2 = ||A||_F^2 (matches the "energy" view in the notes).
    energy = S * S
    total_energy = float(energy.sum())

    if total_energy <= tol:
        # Degenerate image: return raw leading singular values (zeros likely) and ranks 0
        top = S[:p].astype(np.float32, copy=False)
        return np.concatenate([top, np.array([0.0, 0.0], dtype=np.float32)])

    cum_energy = np.cumsum(energy) / total_energy

    # Effective ranks at 95% and 99% energy (1-based indexing)
    r95 = float(np.searchsorted(cum_energy, 0.95) + 1)
    r99 = float(np.searchsorted(cum_energy, 0.99) + 1)

    # Normalize singular values in an energy-consistent way:
    # sigma_i / ||A||_F  (dimensionless, stable scaling)
    frob = np.sqrt(total_energy)
    top = (S[:p] / frob).astype(np.float32, copy=False)

    return np.concatenate([top, np.array([r95, r99], dtype=np.float32)])


def lda_train(X, y):
    """
    Train a two-class LDA model.

    Returns:
      w, tau
    where classifier is: 1 if (w^T x) >= tau else 0.
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

    # Within-class scatter: Sw = sum_{x in class0} (x-mu0)(x-mu0)^T + same for class1
    X0c = X0 - mu0
    X1c = X1 - mu1
    Sw = (X0c.T @ X0c) + (X1c.T @ X1c)

    # Regularization (ridge) to stabilize if Sw is singular/ill-conditioned:
    # Sw_reg = Sw + lam * I  (matches the recommended approach in the project notes)
    d = Sw.shape[0]
    tr = float(np.trace(Sw))
    lam = (1e-6 * tr / d) if tr > 0.0 else 1e-6
    Sw_reg = Sw + lam * np.eye(d, dtype=np.float64)

    # Two-class LDA direction: w ∝ Sw^{-1}(mu1 - mu0), computed via solve (no explicit inverse).
    w = np.linalg.solve(Sw_reg, (mu1 - mu0))

    # Natural threshold: midpoint of projected class means
    m0 = float(w @ mu0)
    m1 = float(w @ mu1)
    tau = 0.5 * (m0 + m1)

    return w, tau


def lda_predict(X, w, threshold):
    X = np.asarray(X, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    scores = X @ w
    return (scores >= threshold).astype(int)


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

    def build_features(X):
        feats = [svd_features(img, p) for img in X]
        return np.vstack(feats)

    Xf_train = build_features(X_train)
    Xf_test = build_features(X_test)

    print("Feature dimension:", Xf_train.shape[1])

    w, threshold = lda_train(Xf_train, y_train)
    y_pred = lda_predict(Xf_test, w, threshold)

    accuracy = np.mean(y_pred == y_test)
    print(f"Example test accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    _example_run()

