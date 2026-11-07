"""Checksum-verified source downloads for the manuscript analyses."""
from pathlib import Path
import hashlib,json,urllib.request,os,time,shutil

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def source_manifest(root):
    return json.loads((Path(root)/'data/sources.json').read_text(encoding='utf-8'))

def select_sources(root,datasets):
    rows=source_manifest(root)
    if datasets:
        unknown=set(datasets)-{x['dataset'] for x in rows}
        if unknown:raise ValueError('Unknown data source(s): '+','.join(sorted(unknown)))
        rows=[x for x in rows if x['dataset'] in datasets]
    return rows

def download_one(row,directory,reserve_bytes=2*1024**3):
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    dest=(directory/row['filename']).resolve()
    if not dest.is_relative_to(directory):raise ValueError('Unsafe manifest filename')
    if dest.is_file():
        if dest.stat().st_size==int(row['bytes']) and sha256(dest)==row['sha256']:
            return dict(source=row['id'],status='REUSED_SHA256_VERIFIED',path=str(dest),downloaded_bytes=0)
        raise RuntimeError('Existing file does not match source; preserve it for diagnosis: '+str(dest))
    if not row['url']:raise RuntimeError('No direct URL; use the recorded official landing page: '+row['landing_page'])
    partial=dest.with_suffix(dest.suffix+'.partial')
    count=partial.stat().st_size if partial.exists() else 0
    if count>int(row['bytes']):raise RuntimeError('Partial file exceeds the registered source size')
    if count==int(row['bytes']):
        if sha256(partial)!=row['sha256']:raise RuntimeError('Complete partial has the wrong checksum; retained unchanged')
        os.replace(partial,dest)
        return dict(source=row['id'],status='RESUMED_SHA256_VERIFIED',path=str(dest),downloaded_bytes=0)
    if shutil.disk_usage(directory).free < int(row['bytes'])-count+reserve_bytes:raise RuntimeError('Insufficient free disk before download')
    start=time.time();initial=count;h=hashlib.sha256()
    if count:
        with partial.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    headers={'User-Agent':'DrugPerturbDB','Accept-Encoding':'identity'}
    if count:headers['Range']=f'bytes={count}-'
    req=urllib.request.Request(row['url'],headers=headers)
    with urllib.request.urlopen(req,timeout=90) as response:
        if count and (response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {count}-')):
            raise RuntimeError('Source does not support the requested resume range; partial retained unchanged')
        with partial.open('ab' if count else 'xb') as out:
            while True:
                block=response.read(8*1024**2)
                if not block:break
                count+=len(block)
                if count>int(row['bytes']):raise RuntimeError('Source exceeds frozen byte count')
                out.write(block);h.update(block)
    if count!=int(row['bytes']) or h.hexdigest()!=row['sha256']:
        raise RuntimeError('Downloaded source checksum/size mismatch; partial retained for diagnosis')
    os.replace(partial,dest)
    return dict(source=row['id'],status='RESUMED_SHA256_VERIFIED' if initial else 'DOWNLOADED_SHA256_VERIFIED',url=row['url'],path=str(dest),bytes=count,sha256=h.hexdigest(),seconds=time.time()-start,downloaded_bytes=count-initial)
