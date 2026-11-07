"""Extract the versioned CIGS 20-uM raw-count tables with member checksums."""
from pathlib import Path
import hashlib,json,tarfile,shutil

R=Path('@WORKSPACE@/FULL_DATASET_FIGURES_V1')
archive=R/'raw/CIGS_TCM20_counts_v1.tar.gz'
dest=R/'raw/cigs_snapshots'
with tarfile.open(archive,'r:gz') as stream:
    manifest=json.load(stream.extractfile('MANIFEST.json'))
    assert manifest['version']=='TCM20-counts-v1'
    expected={x['filename']:x for x in manifest['files']}
    assert set(stream.getnames())==set(expected)|{'MANIFEST.json'}
    dest.mkdir(exist_ok=True)
    for name,entry in expected.items():
        target=dest/name
        assert target.resolve().parent==dest.resolve()
        if target.exists():raise FileExistsError(target)
        member=stream.getmember(name)
        assert member.isfile() and member.size==entry['bytes']
        with stream.extractfile(member) as src,target.open('xb') as out:
            shutil.copyfileobj(src,out,8*1024**2)
        with target.open('rb') as check:
            assert hashlib.file_digest(check,'sha256').hexdigest()==entry['sha256']
    (dest/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
