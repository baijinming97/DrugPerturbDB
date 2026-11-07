from ff_common import *
out={};base=None
for i in range(6):
 with h5py.File(R/'data'/f'CIGS_{i}.h5') as h:
  g=h['gene'].asstr()[:];sg=set(g)
  if base is None:base=sg
  out[str(i)]=dict(n=len(g),unique=len(sg),missing=sorted(base-sg),extra=sorted(sg-base))
dump(R/'inventory/CIGS_GENE_CHECK.json',out);print(out,flush=True)
