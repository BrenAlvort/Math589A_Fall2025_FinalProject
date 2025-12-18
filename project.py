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

    iters = 0
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
    k = int(k)
    if not (1 <= k <= r):
        raise ValueError(f"k must be in [1,{r}]")

    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    Ak = (U[:, :k] * s[:k]) @ Vt[:k, :]

    denom = np.linalg.norm(A, ord="fro")
    rel_err = 0.0 if denom == 0 else float(np.linalg.norm(A - Ak, ord="fro") / denom)
    comp_ratio = float(k * (m + n + 1) / (m * n))
    return Ak, rel_err, comp_ratio


# =========================================================
# 3. Feature extractors you can test locally
# =========================================================
def _svd_spectrum(image):
    A = np.asarray(image, dtype=float)
    if A.ndim != 2:
        raise ValueError("image must be 2D")
    return np.linalg.svd(A, full_matrices=False, compute_uv=False)


def svd_features_cumsum_s(image, p, tol=1e-12):
    """
    "0.7-style": cumulative proportions of singular values:
      feat = [cumsum(s)/sum(s)][:p] + [r_0.9, r_0.95]
    """
    s = _svd_spectrum(image)
    r = s.size
    p = int(p)
    if not (1 <= p <= r):
        raise ValueError(f"p must be in [1,{r}]")

    s_sum = float(np.sum(s))
    if s_sum > tol:
        cum = np.cumsum(s) / s_sum
    else:
        cum = np.zeros_like(s, dtype=float)

    r09 = float(np.searchsorted(cum, 0.90) + 1)
    r095 = float(np.searchsorted(cum, 0.95) + 1)
    return np.hstack((cum[:p], [r09, r095])).astype(float)


def svd_features_cumsum_s2(image, p, tol=1e-12):
    """
    Energy variant: cumulative proportions of squared singular values:
      feat = [cumsum(s^2)/sum(s^2)][:p] + [r_0.9, r_0.95]
    """
    s = _svd_spectrum(image)
    r = s.size
    p = int(p)
    if not (1 <= p <= r):
        raise ValueError(f"p must be in [1,{r}]")

    e = s * s
    e_sum = float(np.sum(e))
    if e_sum > tol:
        cum = np.cumsum(e) / e_sum
    else:
        cum = np.zeros_like(e, dtype=float)

    r09 = float(np.searchsorted(cum, 0.90) + 1)
    r095 = float(np.searchsorted(cum, 0.95) + 1)
    return np.hstack((cum[:p], [r09, r095])).astype(float)


def svd_features_topnorm_s_energy_ranks(image, p, tol=1e-12):
    """
    Standard: normalized top singular values + energy ranks.
      feat = [s1/sum(s),...,sp/sum(s)] + [r_0.9, r_0.95] based on s^2
    """
    s = _svd_spectrum(image)
    r = s.size
    p = int(p)
    if not (1 <= p <= r):
        raise ValueError(f"p must be in [1,{r}]")

    s_sum = float(np.sum(s))
    top = (s[:p] / s_sum) if s_sum > tol else np.zeros(p, dtype=float)

    e = s * s
    e_sum = float(np.sum(e))
    if e_sum > tol:
        cum = np.cumsum(e) / e_sum
        r09 = float(np.searchsorted(cum, 0.90) + 1)
        r095 = float(np.searchsorted(cum, 0.95) + 1)
    else:
        r09 = 0.0
        r095 = 0.0

    return np.concatenate([top, np.array([r09, r095], dtype=float)])


# =========================================================
# Pick ONE feature function for submission API
# (Change this locally after you evaluate.)
# =========================================================
def svd_features(image, p):
    # Default to the simple cumsum(s) style as a baseline.
    return svd_features_cumsum_s(image, p)


# =========================================================
# 4. Two-class LDA: training (stable + label-robust)
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
    Sw_reg = Sw + lam * np.eye(d)

    w = np.linalg.solve(Sw_reg, (mu1 - mu0))

    m0 = float(mu0 @ w)
    m1 = float(mu1 @ w)
    threshold = 0.5 * (m0 + m1)

    if m1 < m0:
        w = -w
        threshold = -threshold

    return w, float(threshold)


# =========================================================
# 5. Two-class LDA: prediction
# =========================================================
def lda_predict(X, w, threshold):
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != w.size:
        raise ValueError("dimension mismatch X and w")
    return (X @ w >= float(threshold)).astype(int)


# =========================================================
# Local: evaluate variants on a held-out split
# =========================================================
def _build_features(X_imgs, p, feat_fn):
    return np.vstack([feat_fn(img, p) for img in X_imgs])


def _acc(y_pred, y_true, y_train_ref):
    classes = np.unique(y_train_ref)
    y_true01 = (y_true == classes.max()).astype(int)
    return float(np.mean(y_pred == y_true01))


def _tune_locally(npz_path="project_data_example.npz", seed=0, val_frac=0.2):
    data = np.load(npz_path)
    X = data["X_train"]
    y = data["y_train"]

    N = X.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    nval = max(1, int(round(val_frac * N)))
    va_idx = perm[:nval]
    tr_idx = perm[nval:]

    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[va_idx], y[va_idx]

    p = min(20, min(X.shape[1], X.shape[2]))

    candidates = [
        ("cumsum(s)", svd_features_cumsum_s),
        ("cumsum(s^2)", svd_features_cumsum_s2),
        ("topnorm(s)+energy ranks", svd_features_topnorm_s_energy_ranks),
    ]

    results = []
    for name, feat_fn in candidates:
        Xf_tr = _build_features(Xtr, p, feat_fn)
        Xf_va = _build_features(Xva, p, feat_fn)

        w, thr = lda_train(Xf_tr, ytr)
        yhat = lda_predict(Xf_va, w, thr)
        acc = _acc(yhat, yva, ytr)
        results.append((name, acc))

    results.sort(key=lambda t: t[1], reverse=True)
    print("Validation results:")
    for name, acc in results:
        print(f"  {name:22s}  acc={acc:.4f}")

    print("\nSet svd_features(...) to your best-performing feature extractor above.")


if __name__ == "__main__":
    # Comment this out for submission; use locally only.
    try:
        _tune_locally("project_data_example.npz", seed=0, val_frac=0.2)
    except Exception as e:
        print("Local tuning skipped:", e)
