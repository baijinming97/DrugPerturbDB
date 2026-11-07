"""Recompute the ten fixed Figure S2 configurations from external features and TRAIN responses."""
import argparse, ast, collections, hashlib, json, os, time, types
from pathlib import Path
import numpy as np
import pandas as pd


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def prepare_targets(workspace, output):
    source = workspace / "4MAN/minimum_restore_v1/retained/orion/baicai/gamma_empirical_v1_20261008T190322Z/data"
    panels = [load_npz(source / f"A_f{f}_TRAIN.npz") for f in range(5)]
    roles = collections.Counter(); groups = {}
    for z in panels:
        d = dict(zip(z["drug_id"].astype(str), z["component"].astype(str)))
        roles.update(d.keys()); groups.update(d)
    sizes = collections.Counter(groups[d] for d in roles)
    output.mkdir(parents=True, exist_ok=True)
    for f, z in enumerate(panels):
        drugs = np.array(sorted(set(z["drug_id"].astype(str))))
        cells = np.array(sorted(set(z["cell_id"].astype(str))))
        lookup = {(c, d): i for i, (c, d) in enumerate(zip(z["cell_id"], z["drug_id"]))}
        ix = np.array([[lookup[c, d] for d in drugs] for c in cells])
        x = z["x"][ix]; y = z["y"][ix].astype(float)
        flat = np.ascontiguousarray(x.transpose(1, 0, 2).reshape(len(drugs), -1))
        _, context = np.unique(flat.view(np.dtype((np.void, flat.dtype.itemsize * flat.shape[1]))).ravel(), return_inverse=True)
        weight = np.array([1 / (len(sizes) * sizes[groups[d]] * roles[d]) for d in drugs])
        np.savez_compressed(output / f"TRAIN_f{f}.npz", drug=drugs, group=np.array([groups[d] for d in drugs]), cell=cells, context=context, mean=y.mean(0), X=x, weight=weight)


def numerical_engine(inputs, targets, work, cap):
    start = time.monotonic(); cache = {}
    def nz(path):
        path = Path(path)
        if path.name == "PAIRED_TRAIN_BETA.npz":
            return {}  # None of the ten T0 configurations uses repeat-derived targets.
        if path.parent.name == "features": path = inputs / path.name
        if path not in cache: cache[path] = load_npz(path)
        return dict(cache[path])
    def check_budget():
        if (time.monotonic() - start) * int(os.environ.get("TAHOE_REPRO_CPUS", "1")) / 3600 > cap:
            raise RuntimeError("Declared CPU-hour cap reached")
    helper = Path(__file__).parent / "helpers"
    ns = dict(np=np, pd=pd, Path=Path, os=os, sys=__import__("sys"), json=json, time=time, hashlib=hashlib,
              collections=collections, R=work, V1=targets, O=work.parent, reg=Path, nz=nz, check_budget=check_budget)
    selected = {"weights", "splits", "corr", "proj", "basis", "group_perm", "preprocess", "gram", "regress", "knn", "summarize"}
    path = helper / "BETA_DISCOVERY_V1/bd_core.py"
    tree = ast.parse(path.read_text())
    body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) or isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in selected]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), ns)
    ns["_BOOTSTRAP_DRAWS"] = {}; ns["c"] = types.SimpleNamespace(**{k: ns[k] for k in selected})
    path = helper / "BETA_DISCOVERY_V2/bd2_models.py"
    tree = ast.parse(path.read_text())
    body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)) and not (isinstance(n, ast.ImportFrom) and n.module == "bd2_common")]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), ns)
    return ns


def moa_plans(trial, inputs, splits):
    mech = load_npz(inputs / "MoA.npz")
    lut = dict(zip(mech["drug"], mech["moa"]))
    moa = np.array([str(lut.get(d, "__missing__")) for d in trial.p["drug"]])
    groups = trial.p["group"]; parent = list(range(trial.n))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for label in [moa, groups]:
        for g in np.unique(label):
            ii = np.flatnonzero(label == g)
            for j in ii[1:]: parent[find(j)] = find(int(ii[0]))
    joint = np.array([str(find(i)) for i in range(trial.n)])
    known = dict(zip(mech["drug"], mech["moa_available"]))
    mask = np.array([bool(known.get(d, False)) for d in trial.p["drug"]])
    trial.av &= mask; ii = np.flatnonzero(mask)
    return [(ii[a], ii[b]) for a, b in splits(joint[ii], 4)]


def run(args):
    inputs = args.inputs.resolve(); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    configs = json.loads((inputs / "CONFIGURATIONS.json").read_text())
    cfg = next(c for c in configs if c["id"] == args.config)
    work = output / cfg["id"]; work.mkdir(exist_ok=True)
    if (work / "SUMMARY.json").exists(): raise FileExistsError(work)
    targets = output / "train_inputs"
    if not all((targets / f"TRAIN_f{f}.npz").exists() for f in range(5)): prepare_targets(args.workspace.resolve(), targets)
    engine = numerical_engine(inputs, targets, work, args.core_hours)
    rows = []; predictions = {}; selected = []
    for seed in [None, 1337, 2026, 3407]:
        for fold in range(5):
            trial = engine["Trial"](fold, dict(cfg))
            plans = moa_plans(trial, inputs, engine["splits"]) if cfg.get("split") == "MoA_known" else None
            res = trial.run(seed, plans); p = trial.p
            full_truth = engine["proj"](p["mean"], p["context"], engine["weights"](p["group"]))
            legacy = engine["corr"](res["raw"], full_truth)
            strict = engine["corr"](res["raw"], res["truth"])
            name = "true" if seed is None else str(seed)
            for j, drug in enumerate(p["drug"]):
                rows.append(dict(config=cfg["id"], seed=name, fold=fold, drug=drug, group=p["group"][j], weight=float(p["weight"][j]), covered=bool(res["available"][j]), value=float(legacy[j]), strict_block_PCC=float(strict[j])))
            selected.extend([dict(seed=name, **s) for s in res["selected"]])
            if seed is None:
                for k in ["raw", "pred", "truth", "available", "block"]: predictions[f"f{fold}_{k}"] = res[k]
                for k in ["drug", "group", "weight"]: predictions[f"f{fold}_{k}"] = p[k]
            pd.DataFrame(rows).to_csv(work / "PER_DRUG.csv", index=False)
            print(cfg["id"], name, fold, flush=True)
    frame = pd.DataFrame(rows); real = frame[(frame.seed == "true") & frame.covered].copy()
    ctrl = frame[(frame.seed != "true") & frame.covered].groupby(["fold", "drug"])[["value", "strict_block_PCC"]].mean()
    real = real.join(ctrl, on=["fold", "drug"], rsuffix="_permuted")
    real["paired_delta"] = real.value - real.value_permuted
    real["strict_paired_delta"] = real.strict_block_PCC - real.strict_block_PCC_permuted
    scores = engine["summarize"](real, ["value", "strict_block_PCC", "value_permuted", "paired_delta", "strict_paired_delta"])
    (work / "SUMMARY.json").write_text(json.dumps(dict(config=cfg, drugs=int(real.drug.nunique()), groups=int(real.group.nunique()), TEST_rows=0, VALID_rows=0, **scores), indent=2))
    (work / "SELECTIONS.json").write_text(json.dumps(selected, indent=2, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item()))
    np.savez_compressed(work / "PREDICTIONS.npz", **predictions)
    print(json.dumps(scores), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--config")
    choice.add_argument("--all", action="store_true", help="Recompute all ten fixed configurations")
    parser.add_argument("--core-hours", type=float, default=3)
    args=parser.parse_args()
    if args.all:
        for config in json.loads((args.inputs / "CONFIGURATIONS.json").read_text()):
            args.config=config["id"]; run(args)
    else: run(args)
