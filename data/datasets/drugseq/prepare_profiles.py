from ff_common import *
import struct
start=time.time();name='DRUGseq';quota();log=(R/'logs/DRUGseq_R.log').open('wb');proc=subprocess.Popen(['Rscript','--vanilla',str(R/'code/drugseq_stream.R')],stdout=subprocess.PIPE,stderr=log)
def exact(n):
 b=bytearray()
 while len(b)<n:
  v=proc.stdout.read(n-len(b))
  if not v:raise RuntimeError(('R_stream_truncated',len(b),n,proc.poll()))
  b.extend(v)
 return b
try:
 n,g=struct.unpack('<ii',exact(8));genes=(R/'data/DRUGseq_GENE_IDS.txt').read_text().splitlines();assert len(genes)==g
 md=pd.read_csv(R/'data/DRUGseq_STREAM_META.tsv',sep='\t',dtype=str,keep_default_na=False);assert len(md)==n
 md=md.rename(columns={'cmpd_sample_id':'drug','cell_line_name':'cell','concentration':'dose','hours_post_treatment':'time','plate_barcode':'plate','biosample_id':'physical_unit'});md['dose']=md.dose.astype(float);md['time']=md.time.astype(float);md['is_control']=md.well_type.eq('RC');md['cell']='CVCL_0042';md['control_key']=md.plate;md['repeat_type']='cross_plate';md['panel']='MoABox'
 outmeta=pd.concat([md.assign(normalization='logCPM1'),md.assign(normalization='logCPM100')],ignore_index=True);seen=np.zeros(n,bool);total=np.zeros(n,np.int64)
 with newh5(name,outmeta,genes,'drugseq') as h,h5py.File(R/'data/DRUGseq_COUNTS.h5','w') as hc:
  counts=hc.create_dataset('counts',shape=(n,g),dtype='i4',chunks=(16,min(g,4096)),compression='gzip',compression_opts=1);hc.create_dataset('gene',data=np.asarray(genes,object),dtype=h5py.string_dtype())
  while seen.sum()<n:
   nr=struct.unpack('<i',exact(4))[0];ix=np.frombuffer(exact(nr*4),dtype='<i4');ct=np.frombuffer(exact(nr*g*4),dtype='<i4').reshape(nr,g);assert not seen[ix].any();seen[ix]=True;assert np.array_equal(ix,np.arange(ix[0],ix[-1]+1));sl=slice(int(ix[0]),int(ix[-1])+1);counts[sl]=ct;tot=ct.sum(1,dtype=np.int64);total[ix]=tot;ci=md.iloc[ix].is_control.to_numpy()&(tot>0);assert ci.any()
   for a,pc in enumerate([1,100]):
    expr=np.log2(ct.astype(np.float64)/np.maximum(tot,1)[:,None]*1e6+pc);delta=expr-expr[ci].mean(0);sl2=slice(sl.start+a*n,sl.stop+a*n);h['post'][sl2]=expr;h['delta'][sl2]=delta
   print('plate',md.iloc[int(ix[0])].plate,'completed',int(seen.sum()),n,flush=True)
  hc.create_dataset('total',data=total)
 assert proc.wait()==0;outmeta['valid_profile']=np.tile(total>0,2);outmeta['library_total']=np.tile(total,2);outmeta['row']=np.arange(len(outmeta));outmeta.to_parquet(R/'data/DRUGseq_META.parquet',index=False)
 finish(name,start,dict(raw_sha256=sha(R/'raw/Exp_gzip.RData'),physical_wells=n,drugs=md.loc[~md.is_control,'drug'].nunique(),native_genes=g,zero_library_wells=int((total==0).sum()),controls='all nonempty RC DMSO wells in same physical plate',all_metadata_wells_found=True))
except BaseException:
 proc.terminate();proc.wait();raise
finally:log.close()
