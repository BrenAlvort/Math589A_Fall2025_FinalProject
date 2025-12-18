import numpy as np

def svd_features(image, p, tol=1e-12):
    A = np.asarray(image, dtype=np.float64)
    S = np.linalg.svd(A, full_matrices=False, compute_uv=False)

    e = S * S
    tot = float(e.sum())
    if tot <= tol:
        return np.zeros(p + 2, dtype=np.float32)

    cum = np.cumsum(e) / tot

    # Smooth p-length signature: cumulative energy at 1..p
    lead = cum[:p].astype(np.float32, copy=False)

    r95 = float(np.searchsorted(cum, 0.95) + 1)
    r99 = float(np.searchsorted(cum, 0.99) + 1)

    return np.concatenate([lead, np.array([r95, r99], dtype=np.float32)])
