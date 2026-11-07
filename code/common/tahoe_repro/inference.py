"""CPU inference for the frozen control-expression baseline and G5 decoder.

Load only trusted model files verified against data/config/models.json."""
from pathlib import Path
import gzip
import hashlib
import pickle
import numpy as np


def verify_file(path, expected_sha256):
    with Path(path).open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected_sha256:
        raise ValueError(f'Model SHA256 mismatch: {path}')


def predict_control_only(path, x, *, expected_sha256, threads=1):
    import torch
    from torch import nn
    verify_file(path, expected_sha256)
    if not 1 <= threads <= 8:
        raise ValueError('CPU threads must be in [1, 8]')
    torch.set_num_threads(threads)

    class BasalMLP(nn.Module):
        def __init__(self):
            super().__init__()
            layers, width = [], 978
            for _ in range(3):
                layers.extend([nn.Linear(width, 2048), nn.ReLU(), nn.Dropout(.1)])
                width = 2048
            layers.append(nn.Linear(width, 978))
            self.net = nn.Sequential(*layers)

        def forward(self, inputs):
            return self.net(inputs)

    ck = torch.load(path, map_location='cpu', weights_only=True)
    if ck['seed'] != 1337:
        raise ValueError('The paper baseline is seed1337')
    net = BasalMLP()
    net.load_state_dict(ck['model'], strict=True)
    net.eval()
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] != 978 or not np.isfinite(x).all():
        raise ValueError('Expected finite author-normalized control expression on the frozen 978-gene axis')
    unique, inverse = np.unique(x.reshape(-1, 978), axis=0, return_inverse=True)
    with torch.inference_mode():
        y = net(torch.from_numpy(unique)).numpy()
    return y[inverse].reshape(x.shape).astype(np.float64)


def predict_g5(path, beta, *, expected_sha256):
    """Return raw C x D x G interactions and frozen TRAIN-calibrated lambda.

    beta must follow the frozen gene order. Project the raw predictions in the
    evaluation panel before adding them to a baseline. True beta is an oracle
    input, not a deployable drug feature.
    """
    verify_file(path, expected_sha256)
    with gzip.open(path, 'rb') as stream:
        fitted, shrinkage = pickle.load(stream)
    beta = np.asarray(beta, np.float64)
    if beta.ndim != 2 or beta.shape[1] != 978 or not np.isfinite(beta).all():
        raise ValueError('Expected a finite drug x 978 beta matrix')
    h = fitted
    x = (beta - h['mu']) @ h['v'].T / h['scale']
    x = np.column_stack([x, np.ones(len(x))])  # log10(5 uM / 5 uM) + 1
    scores = (x - h['xm']) @ h['coef'] + h['ym']
    result = (scores @ h['U']).reshape(len(beta), 47, 978).transpose(1, 0, 2)
    return result, float(shrinkage)
