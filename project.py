import numpy as np

# =========================================================
# 1. Power method for dominant eigenpair
# =========================================================
def power_method(A, x0, maxit, tol):
    A = np.asarray(A, dtype=float)
    x = np.asarray(x0, dtype=float).reshape(-1)

    lam0 = float(x @ (A @ x))
    iters = 0

    for iters in range(1, int(maxit) + 1):
        y = A @ x
        ny = np.linalg.norm(y)
        if ny == 0:
            return 0.0, x, iters

        x = y / ny
        lam = float(x @ (A @ x))

        # relative change stopping
        if abs(lam) > 0:
            err = abs(lam - lam0) / abs(lam)
        else:
            err = abs(lam - lam0)

        if err < tol:
            return lam, x, iters

        lam0 = lam

    return lam0, x, iters


# =========================================================
# 2. Rank-k image compression via SVD
# =========================================================
def svd_compress(image, k):
    A = np.asarray(image, dtype=float)
    U, s, Vh = np.linalg.svd(A, full_matrices=False)

    k = int(k)
    Ak = (U[:, :k] * s[:k]) @ Vh[:k, :]

    denom = np.linalg.norm(A, ord="fro")
    rel_err = 0.0 if denom == 0 else float(np.linalg.norm(A - Ak, ord="fro") / denom)

    m, n = A.shape
    comp_ratio = float(k * (m + n + 1) / (m * n))
    return Ak, rel_err, comp_ratio


# =========================================================
# 3. Roughness features on blocks (simple)
# =========================================================
def block_roughness_features(image, block_size=8):
    A = np.asarray(image, dtype=float)
    H, W = A.shape

    rough = []
    b = int(block_size)

    # use full blocks only
    Hc = (H // b) * b
    Wc = (W // b) * b

    for i in range(0, Hc, b):
        for j in range(0, Wc, b):
            B = A[i:i+b, j:j+b]
            dx = B[:, 1:] - B[:, :-1]
            dy = B[1:, :] - B[:-1, :]
            r = np.mean(dx * dx) + np.mean(dy * dy)
            rough.append(r)

    if len(rough) == 0:
        return np.array([0.0, 0.0, 0.0], dtype=float)

    rough = np.array(rough, dtype=float)
    return np.array([rough.mean(), rough.std(), rough.max()], dtype=float)


# =========================================================
# 4. SVD-based features (+ roughness)
# =========================================================
def svd_features(image, p, tol=1e-12):
    A = np.asarray(image, dtype=float)
    s = np.linalg.svd(A, full_matrices=False, compute_uv=False)

    p = int(p)
    ssum = float(np.sum(s))
    top = (s[:p] / ssum) if ssum > tol else np.zeros(p, dtype=float)

    e = s * s
    etot = float(np.sum(e))
    if etot > tol:
        c = np.cumsum(e) / etot
        r95 = float(np.searchsorted(c, 0.95) + 1)
        r99 = float(np.searchsorted(c, 0.99) + 1)
    else:
        r95 = 0.0
        r99 = 0.0

    rough = block_roughness_features(A, block_size=8)
    return np.concatenate([top, np.array([r95, r99], dtype=float), rough])


# =========================================================
# 5. LDA train / predict (simple + small ridge)
# =========================================================
def lda_train(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).reshape(-1)

    # assumes y is 0/1 (like most project specs)
    X0 = X[y == 0]
    X1 = X[y == 1]

    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)

    X0c = X0 - mu0
    X1c = X1 - mu1
    Sw = (X0c.T @ X0c) + (X1c.T @ X1c)

    d = Sw.shape[0]
    lam = 1e-6 * (np.trace(Sw) / d) if np.trace(Sw) > 0 else 1e-6
    Sw = Sw + lam * np.eye(d)

    w = np.linalg.solve(Sw, (mu1 - mu0))

    m0 = float(mu0 @ w)
    m1 = float(mu1 @ w)
    thr = 0.5 * (m0 + m1)

    # keep direction consistent
    if m1 < m0:
        w = -w
        thr = -thr

    return w, float(thr)


def lda_predict(X, w, threshold):
    X = np.asarray(X, dtype=float)
    return (X @ w >= float(threshold)).astype(int)


# =========================================================
# Local smoke test (optional)
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

    p = min(32, min(X_train.shape[1], X_train.shape[2]))

    Xf_train = np.vstack([svd_features(img, p) for img in X_train])
    Xf_test = np.vstack([svd_features(img, p) for img in X_test])

    w, thr = lda_train(Xf_train, y_train)
    y_pred = lda_predict(Xf_test, w, thr)

    acc = float(np.mean(y_pred == y_test))
    print(f"Example test accuracy: {acc:.3f}")


if __name__ == "__main__":
    _example_run()
