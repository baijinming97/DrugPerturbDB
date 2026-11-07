from ff_common import *
import urllib.request,gzip,csv
items=[('GPL571.annot.gz','https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL571/annot/GPL571.annot.gz'),('compoundinfo_beta.txt','https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/compoundinfo_beta.txt'),('cellinfo_beta.txt','https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/cellinfo_beta.txt')]
receipt=[]
for name,url in items:
 p=R/'raw'/name
 try:
  if p.exists():receipt.append(dict(name=name,url=url,bytes=p.stat().st_size,sha256=sha(p),reuse=True));continue
  with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=60) as h:size=int(h.headers.get('Content-Length',0))
  assert 0<size<25*2**20,(name,size)
  with urllib.request.urlopen(url,timeout=60) as h,p.open('wb') as b:
   data=h.read(size+1);assert len(data)==size;b.write(data)
  receipt.append(dict(name=name,url=url,bytes=size,sha256=sha(p)))
 except Exception as e:receipt.append(dict(name=name,url=url,error=repr(e)))
dump(R/'receipts/EXTRA_METADATA_DOWNLOADS.json',receipt)
out={}
with gzip.open(R/'raw/GSE116436_series_matrix.txt.gz','rt') as f:
 for line in f:
  if line.startswith('!series_matrix_table_begin'):break
  if line.startswith('!Sample_'):
   vals=next(csv.reader([line],delimiter='\t'));out.setdefault(vals[0],[]).append({'first':vals[1:5],'unique_count':len(set(vals[1:])), 'unique_head':list(dict.fromkeys(vals[1:]))[:30] if len(set(vals[1:]))<100 else []})
dump(R/'inventory/NCI_HEADERS.json',out)
