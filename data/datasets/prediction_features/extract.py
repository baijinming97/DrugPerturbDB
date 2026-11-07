"""Prepare the fixed external features for the ten Figure S2 comparisons."""
from pathlib import Path
import json,hashlib,tarfile
root=Path('@WORKSPACE@')
archive=root/'FULL_DATASET_FIGURES_V1/raw/Exploratory_features_v1.tar.gz'
destination=root/'4MAN/EXPLORATORY/features'
with tarfile.open(archive,'r:gz') as stream:
    manifest=json.load(stream.extractfile('MANIFEST.json'))
    expected={f['file']:f for f in manifest['files']}
    assert set(stream.getnames())==set(expected)|{'MANIFEST.json','CONFIGURATIONS.json'}
    if destination.exists() and any(destination.iterdir()):raise FileExistsError(destination)
    destination.mkdir(parents=True,exist_ok=True)
    for member in stream.getmembers():
        assert member.isfile() and Path(member.name).name==member.name
        content=stream.extractfile(member).read()
        if member.name in expected:
            item=expected[member.name]
            assert len(content)==item['bytes'] and hashlib.sha256(content).hexdigest()==item['sha256']
        (destination/member.name).write_bytes(content)
