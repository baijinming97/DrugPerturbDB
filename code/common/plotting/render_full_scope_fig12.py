"""Plot within-source repeatability and cross-source concordance in Figures 1-2."""
from pathlib import Path
import sys,os,json,hashlib,re
R=Path(__file__).resolve().parents[1];V=R.parent;OLD=R/'updated_inputs';F=R/'figures'
os.environ['MPLCONFIGDIR']=str(R/'code'/'mpl_fig12')
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors,patches,font_manager
from matplotlib.lines import Line2D
from matplotlib.text import Text
INK='#192D3A';TEAL='#147D92';AMBER='#D69B36';PURPLE='#7B66AD';GRAY='#667D8C';LIGHT='#F3F6F8';RULE='#DEE5E9';WIDTH=180/25.4
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#B9C7D0','axes.linewidth':.65,'savefig.facecolor':'white','axes.unicode_minus':False,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'xtick.labelsize':8,'ytick.labelsize':8.5,'hatch.linewidth':.4})
F.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
names=['fig1a_expression.csv','fig1b_response.csv','fig2d_cell_averaged.csv','fig2c_complete_strict.csv','fig2a_direct.csv','fig2b_auc.csv','fig2_multiview_display.csv']
src=[OLD/'figures'/n for n in names];initial={str(p):sha(p) for p in src}
datasets=['Tahoe','L1000_L3','DRUGseq','CIGS','sciPlex3','MIXseq']
labels={'Tahoe':'Tahoe','L1000_L3':'L1000 L3','DRUGseq':'DRUG-seq','CIGS':'CIGS','sciPlex3':'sciPlex3','MIXseq':'MIX-seq'}
display=[];geometry={}
def save(fig,stem):
 fig.canvas.draw();renderer=fig.canvas.get_renderer();used=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
 geometry[stem]={'width_mm':180,'height_mm':float(fig.get_size_inches()[1]*25.4),'minimum_text_pt':min(t.get_fontsize() for t in used)}
 assert geometry[stem]['minimum_text_pt']>=7.2
 fw,fh=fig.canvas.get_width_height();outside=[]
 for t in used:
  b=t.get_window_extent(renderer)
  if b.x0<-.5 or b.y0<-.5 or b.x1>fw+.5 or b.y1>fh+.5:outside.append(t.get_text())
 geometry[stem]['text_outside_canvas']=outside;assert not outside,(stem,outside)
 if stem.startswith('fig1'):
  axisbox=fig.axes[0].xaxis.label.get_window_extent(renderer);legendbox=fig.legends[0].get_window_extent(renderer)
  gap=(axisbox.y0-legendbox.y1)*72/fig.dpi
  geometry[stem]['axis_label_to_legend_gap_pt']=float(gap);assert gap>=3.0,(stem,gap)
 for ext in ['svg','png','pdf']:fig.savefig(F/f'{stem}.{ext}',dpi=300,facecolor='white')
 plt.close(fig)
def panel(fig,letter,title,x=.025,y=.947):
 fig.text(x,y,letter,weight='bold',fontsize=12);fig.text(x+.043,y,title,weight='bold',fontsize=10.5)
def interval(ax,x,y,lo,hi,color,marker='o',small=False,size=4.1):
 if np.isfinite(lo):ax.plot([lo,hi],[y,y],color=color,lw=1.05,solid_capstyle='round',zorder=3)
 ax.plot(x,y,marker=marker,ms=size,mfc='white' if small else color,mec=color,mew=.9,ls='none',zorder=4)

for stem,panel_id,title in [('fig1a_expression','A','Normalized treated expression'),('fig1b_response','B','Treatment-induced response')]:
 t=pd.read_csv(OLD/'figures'/f'{stem}.csv',float_precision='round_trip');t=t[t.normalization!='logCPM100'];assert len(t)==12
 fig=plt.figure(figsize=(WIDTH,3.62));panel(fig,panel_id,title)
 ax=fig.add_axes([.31,.300,.445,.515]);ax.set_facecolor('none');ax.set_xlim(-.02,1.02);ax.set_ylim(5.55,-.55)
 ax.set_xticks([0,.25,.5,.75,1],['0','0.25','0.50','0.75','1.00']);ax.set_yticks([])
 ax.grid(axis='x',color=RULE,lw=.6);ax.set_axisbelow(True);ax.spines['left'].set_visible(False)
 ax.tick_params(axis='x',length=3,width=.6);ax.set_xlabel('Replicate PCC',labelpad=5,fontsize=8.5)
 for x,txt,ha in [(.025,'SOURCE','left'),(.268,'DRUG IDs','right'),(.842,'NATIVE','center'),(.946,'COMMON','center')]:fig.text(x,.856,txt,fontsize=7.5,color=GRAY,ha=ha,weight='bold')
 fig.add_artist(Line2D([.025,.98],[.84,.84],transform=fig.transFigure,color=RULE,lw=.7))
 for i,nm in enumerate(datasets):
  ss=t[t.dataset==nm];n=int(ss.drugs.iloc[0]);yf=.300+.515*(5.55-i)/6.10
  if i%2==1:fig.add_artist(patches.Rectangle((.025,yf-.039),.955,.078,transform=fig.transFigure,facecolor=LIGHT,edgecolor='none',zorder=-1))
  fig.text(.025,yf,labels[nm],va='center',fontsize=8.8);fig.text(.268,yf,f'{n:,}',ha='right',va='center',fontsize=8.2,color='#657A87')
  for axis,offset,col,marker,xnum in [('native',-.12,GRAY,'o',.842),('common',.12,TEAL,'D',.946)]:
   z=ss[ss.axis==axis].iloc[0];interval(ax,z.pcc,i+offset,z.low,z.high,col,marker,n<5,3.9)
   fig.text(xnum,yf,f'{z.pcc:.3f}',ha='center',va='center',fontsize=8.7,color=col,weight='bold' if axis=='common' else 'normal')
   display.append(dict(stem=stem,dataset=nm,axis=axis,estimate=z.pcc,low=z.low,high=z.high,drugs=n,genes=int(z.genes)))
 fig.legend(handles=[Line2D([],[],color=GRAY,marker='o',lw=1,label='Native measured genes',ms=4),Line2D([],[],color=TEAL,marker='D',lw=1,label='Common 869 genes',ms=4)],loc='lower left',bbox_to_anchor=(.302,.095),ncol=2,frameon=False,fontsize=8,handlelength=1.5,columnspacing=1.6)
 fig.text(.025,.080,'95% intervals · Identical repeat pairs on both gene axes · Open markers: n < 5',fontsize=7.6,color='#657A87')
 fig.text(.025,.038,'Tahoe: all eligible cross-plate repeats. MIX-seq: one drug; no drug-level interval.',fontsize=7.6,color='#657A87')
 save(fig,stem)

mean=pd.read_csv(src[2],float_precision='round_trip');strict=pd.read_csv(src[3],float_precision='round_trip')
retr=pd.read_csv(src[4],float_precision='round_trip');auc=pd.read_csv(src[5],float_precision='round_trip');pd.testing.assert_frame_equal(retr,auc)
assert len(mean)==len(strict)==15 and (strict.drugs>0).sum()==8
observed=pd.concat([mean,strict]);observed=observed[observed.drugs>0];assert observed.estimate.min()>=-.02 and observed.estimate.max()<=.5
cmap=colors.LinearSegmentedColormap.from_list('response_pcc',[(0,'#DACEEA'),(.02/.52,'#F1F6F7'),(.25,'#C1DADE'),(.50,'#6EAEBB'),(.75,TEAL),(1,'#09556A')]);norm=colors.Normalize(-.02,.5,clip=False)
def heatmap(ax,table,panel_id):
 for i,a in enumerate(datasets):
  for j,b in enumerate(datasets):
   if i<=j:continue
   row=table[((table.a==a)&(table.b==b))|((table.a==b)&(table.b==a))].iloc[0];n=int(row.drugs);v=float(row.estimate);y=i-1
   if n==0:
    ax.add_patch(patches.Rectangle((j-.48,y-.48),.96,.96,facecolor='#E8ECF0',edgecolor='none'))
    ax.text(j,y,'-',ha='center',va='center',fontsize=9,color='#9AA7B1')
   else:
    face=cmap(norm(v));light=v<.27;txtcol=INK if light else 'white';subcol=INK if light else 'white'
    ax.add_patch(patches.Rectangle((j-.48,y-.48),.96,.96,facecolor=face,edgecolor='#97AAB4' if n<5 else 'none',lw=.5,hatch='///' if n<5 else None))
    txt=f'{v:.3f}' if abs(v)>=.0005 else f'{v:+.4f}'
    ax.text(j,y-.13,txt,ha='center',va='center',fontsize=8 if abs(v)<.0005 else 9.4,weight='bold',color=txtcol)
    ax.text(j,y+.24,f'n={n:,}',ha='center',va='center',fontsize=7.4,color=subcol)
   display.append(dict(stem='fig2_ab_similarity',panel=panel_id,a=row.a,b=row.b,metric='pcc',estimate=row.estimate,low=row.low,high=row.high,drugs=n,groups=row.groups))
 ax.set_xlim(-.5,4.5);ax.set_ylim(4.5,-.5);ax.set_aspect('equal')
 ax.set_xticks(range(5),[labels[x] for x in datasets[:-1]],rotation=42,ha='right',fontsize=8);ax.set_yticks(range(5),[labels[x] for x in datasets[1:]],fontsize=8);ax.tick_params(length=0,pad=5)
 for sp in ax.spines.values():sp.set_visible(False)

fig=plt.figure(figsize=(WIDTH,4.63));panel(fig,'A','Cell-averaged responses',.025,.947);panel(fig,'B','Matched cell lines',.525,.947)
fig.text(.025,.900,'869 genes  ·  24 h  ·  dose ratio ≤ 2',fontsize=8,color='#657A87')
ax=fig.add_axes([.125,.335,.335,.335*WIDTH/4.63]);bx=fig.add_axes([.625,.335,.335,.335*WIDTH/4.63]);heatmap(ax,mean,'A');heatmap(bx,strict,'B')
cax=fig.add_axes([.32,.148,.40,.018]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal',ticks=[0,.1,.2,.3,.4,.5]);cb.solids.set_rasterized(False);cb.outline.set_visible(False);cb.ax.tick_params(length=2,labelsize=7.5,pad=3);cb.set_label('Same-drug PCC',fontsize=8,labelpad=4)
fig.text(.025,.038,'n = matched drugs',fontsize=7.8,color='#657A87')
fig.legend(handles=[patches.Patch(facecolor='#E8ECF0',edgecolor='none',label='No eligible match'),patches.Patch(facecolor='white',edgecolor='#97AAB4',hatch='///',label='n < 5; descriptive')],loc='lower right',bbox_to_anchor=(.98,.015),ncol=2,frameon=False,fontsize=7.8,handlelength=1.5,columnspacing=1.4)
save(fig,'fig2_ab_similarity')

def series(metric):return retr[retr.metric==metric].set_index('b').loc[datasets[1:]].reset_index()
fig=plt.figure(figsize=(WIDTH,3.55));panel(fig,'C','Drug-specific agreement',.025,.947);panel(fig,'D','Drug retrieval',.757,.947)
fig.text(.025,.893,'Tahoe to external sources',fontsize=8,color='#657A87')
ax=fig.add_axes([.185,.295,.225,.49]);dx=fig.add_axes([.47,.295,.225,.49]);rx=fig.add_axes([.765,.295,.215,.49])
same,wrong,diff,ret=map(series,['same_pcc','wrong_pcc','pcc_difference','auc'])
for q in [ax,dx,rx]:
 for y in [1,3]:q.axhspan(y-.48,y+.48,color=LIGHT,zorder=0)
 q.set_ylim(4.55,-.55);q.grid(axis='x',color=RULE,lw=.6);q.set_axisbelow(True);q.spines['left'].set_visible(False);q.tick_params(axis='y',length=0,pad=7);q.tick_params(axis='x',length=3,width=.6)
for i,(sr,wr,dr,ar) in enumerate(zip(same.itertuples(),wrong.itertuples(),diff.itertuples(),ret.itertuples())):
 small=int(sr.drugs)<5;ax.plot([wr.estimate,sr.estimate],[i+.13,i-.13],color='#CAD5DC',lw=.8,zorder=1)
 interval(ax,sr.estimate,i-.13,sr.low,sr.high,TEAL,'o',small,4.2);interval(ax,wr.estimate,i+.13,wr.low,wr.high,GRAY,'s',small,3.7)
 interval(dx,dr.estimate,i,dr.low,dr.high,TEAL,'o',small,4.2);interval(rx,ar.estimate,i,ar.low,ar.high,TEAL,'o',small,4.2)
 for metric,r in [('same_pcc',sr),('wrong_pcc',wr),('pcc_difference',dr),('auc',ar)]:display.append(dict(stem='fig2_cd_discrimination',panel='D' if metric=='auc' else 'C',a=r.a,b=r.b,metric=metric,estimate=r.estimate,low=r.low,high=r.high,drugs=int(r.drugs),groups=r.groups))
ax.set_yticks(range(5),[f'{labels[r.b]}\nn={int(r.drugs):,}' for r in same.itertuples()],fontsize=8)
for q in [dx,rx]:q.set_yticks(range(5),['']*5)
for q in [ax,dx]:q.set_xlim(-.065,.455);q.set_xticks([0,.2,.4]);q.axvline(0,color='#9BABBA',ls='--',lw=.7)
ax.set_title('Same vs. wrong drug',fontsize=8.6,pad=7);dx.set_title('Same − wrong',fontsize=8.6,pad=7);rx.set_title('P(same > wrong)',fontsize=8.6,pad=7)
ax.set_xlabel('PCC',fontsize=8);dx.set_xlabel('PCC difference',fontsize=8);rx.set_xlim(.47,1.025);rx.set_xticks([.5,.75,1]);rx.set_xlabel('Retrieval AUC',fontsize=8);rx.axvline(.5,color='#9BABBA',ls='--',lw=.7)
fig.legend(handles=[Line2D([],[],marker='o',color=TEAL,label='Same drug',lw=1,ms=4),Line2D([],[],marker='s',color=GRAY,label='Wrong drug',lw=1,ms=4)],loc='lower left',bbox_to_anchor=(.178,.115),ncol=2,frameon=False,fontsize=8,handlelength=1.5)
fig.text(.78,.147,'Chance = 0.5',fontsize=7.8,color='#657A87');fig.text(.025,.043,'95% intervals  ·  Open markers: n < 5  ·  n = eligible query drugs',fontsize=7.8,color='#657A87')
save(fig,'fig2_cd_discrimination')


assert all(sha(Path(p))==h for p,h in initial.items())
pd.DataFrame(display).to_csv(R/'FIG12_DISPLAY_VALUES.csv',index=False)
(R/'VISUAL_FIG12_QA.json').write_text(json.dumps(dict(status='RENDERED',physical_geometry=geometry,source_hashes=initial,descriptive_scope='Full eligible Tahoe; model evaluation unchanged'),indent=2))
print('Rendered Figure 1 and Figure 2')

