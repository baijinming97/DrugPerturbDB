"""Balanced-panel model metrics. Descriptive unbalanced panels use statistics.py."""
import numpy as np


def decompose(x, drug_weights):
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(drug_weights, dtype=np.float64)
    if x.ndim != 3 or len(w) != x.shape[1] or np.any(w <= 0):
        raise ValueError('Expected cell x drug x gene and positive drug weights')
    w = w / w.sum()
    cell_mean = np.einsum('cdg,d->cg', x, w)
    mu = cell_mean.mean(axis=0)
    beta = x.mean(axis=0) - mu
    return dict(mu=mu, alpha=cell_mean-mu, beta=beta,
                gamma=x-cell_mean[:, None]-beta[None])


def row_pcc(x, y):
    a, b = np.asarray(x, np.float64), np.asarray(y, np.float64)
    a, b = a-a.mean(-1, keepdims=True), b-b.mean(-1, keepdims=True)
    an, bn = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    return np.divide((a*b).sum(-1), an*bn,
                     out=np.full(an.shape, np.nan), where=(an>1e-6)&(bn>1e-6))


def panel_metrics(prediction, truth, weights):
    w = np.asarray(weights, float); w = w/w.sum()
    pr, tr = decompose(prediction, w), decompose(truth, w)
    def mean_rows(v):
        ww = np.broadcast_to(w/len(v), v.shape)
        good = np.isfinite(v)
        return float(np.sum(np.where(good, v, 0)*ww)/np.sum(ww*good)) if good.any() else float('nan')
    def mse(a, b):
        return float(np.einsum('cdg,d->', (a-b)**2, w)/(a.shape[0]*a.shape[2]))
    return dict(full_PCC=mean_rows(row_pcc(prediction, truth)),
                full_MSE=mse(prediction, truth),
                gamma_PCC=mean_rows(row_pcc(pr['gamma'], tr['gamma'])),
                gamma_MSE=mse(pr['gamma'], tr['gamma']))
