"""Read registered, trusted pickle models, with optional gzip compression."""
from pathlib import Path
import pickle,gzip,hashlib
def model_bytes(path):
 p=Path(path)
 if not p.exists() and Path(str(p)+'.gz').exists():p=Path(str(p)+'.gz')
 return gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes()
def load_model(path):return pickle.loads(model_bytes(path))
def logical_model_sha(path):return hashlib.sha256(model_bytes(path)).hexdigest()
