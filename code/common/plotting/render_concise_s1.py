"""Plot normalization sensitivity and matched-support repeatability in Figure S1."""
from pathlib import Path
import sys,os,json,hashlib,re
R=Path(__file__).resolve().parents[1];V=R.parent;E=R/'evidence_inputs';F=R/'figures';F.mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR']=str(R/'code/mpl_concise_s1')
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.patches import Rectangle
from matplotlib import font_manager
INK='#192D3A';TEAL='#147D92';AMBER='#D69B36';BLUE='#7AA6C2';GRAY='#667D8C';LIGHT='#F3F6F8';RULE='#DCE4E9'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.edgecolor':'#B5C3CC','axes.linewidth':.55,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'xtick.labelsize':7.5,'ytick.labelsize':8,'savefig.facecolor':'white'})
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def jvalue(v):
    if isinstance(v,(np.integer,np.floating)):v=v.item()
    return None if isinstance(v,float) and not np.isfinite(v) else v
paths=[E/'SUPPLEMENT_REVIEW_V2_20261102/data/S2.csv',E/'PAPER_TWO_CHECKS_V1/results/exp1/MATCHED_RELIABILITY.csv']
hashes={str(p.relative_to(R)):sha(p) for p in paths}
norm=pd.read_csv(paths[0],float_precision='round_trip');match=pd.read_csv(paths[1],float_precision='round_trip')
norm=norm[norm.axis.eq('common')].copy();assert len(norm)==8 and set(norm.genes)=={869}
match=match[match.axis.eq('primary869')&match.pseudocount.eq(1)&match.metric.isin(['rAA','rBB','rAB'])].copy()
assert len(match)%3==0
NPAIR=len(match)//3
names={'Tahoe':'Tahoe','L1000_L3':'L1000 L3','DRUGseq':'DRUG-seq','CIGS':'CIGS','sciPlex3':'sciPlex3','MIXseq':'MIX-seq'}
records=[]
fig=plt.figure(figsize=(180/25.4,6.05))
def text(x,y,s,size=8,color=INK,weight='normal',**kw):return fig.text(x,y,s,fontsize=size,color=color,weight=weight,**kw)
def panel(y,letter,title):text(.04,y,letter,12,weight='bold');text(.08,y,title,10.2,weight='bold')
def style(ax,xlim,ticks,label):
    ax.set_facecolor('none');ax.set_xlim(*xlim);ax.set_xticks(ticks);ax.set_yticks([]);ax.grid(axis='x',color=RULE,lw=.5);ax.set_axisbelow(True);ax.tick_params(axis='x',length=3,width=.55);ax.set_xlabel(label,fontsize=8,labelpad=4);ax.axvline(0,color='#A2AFB8',lw=.65,zorder=1)
def point(ax,x,y,lo,hi,col,marker='o',filled=True):
    if np.isfinite(lo) and np.isfinite(hi):
        ax.plot([lo,hi],[y,y],lw=1.05,color=col,zorder=3)
        ax.plot([lo,hi],[y,y],ls='none',marker='|',ms=3.8,mew=.7,color=col,zorder=3)
    ax.plot(x,y,ls='none',marker=marker,ms=4.2,mew=.85,mec=col,mfc=col if filled else 'white',zorder=4)
panel(.958,'A','Processing sensitivity')
text(.04,.925,'Identical repeat pairs · primary 869-gene axis',7.7,GRAY)
ax=fig.add_axes([.34,.643,.42,.220]);ax.set_ylim(3.5,-.5);style(ax,(0,.60),[0,.2,.4,.6],'Response replicate PCC')
fig.legend(handles=[Line2D([],[],color=TEAL,marker='o',ms=4,lw=1,label='log2(CPM + 1)'),Line2D([],[],color=AMBER,marker='D',ms=4,lw=1,label='log2(CPM + 100)')],loc='lower left',bbox_to_anchor=(.326,.881),frameon=False,ncol=2,fontsize=7.6,handlelength=1.4,columnspacing=1.25)
text(.04,.884,'SOURCE / DRUG IDs',7.4,GRAY,weight='bold');text(.855,.884,'+1',7.4,TEAL,weight='bold',ha='center');text(.95,.884,'+100',7.4,AMBER,weight='bold',ha='center')
for y,d in enumerate(['DRUGseq','CIGS','sciPlex3','MIXseq']):
    block=norm[norm.dataset.eq(d)].set_index('normalization');assert len(block)==2 and block.drugs.nunique()==1
    n=int(block.drugs.iloc[0]);yf=.643+.220*(3.5-y)/4
    if y%2==0:fig.add_artist(Rectangle((.04,yf-.025),.93,.050,transform=fig.transFigure,fc=LIGHT,ec='none',zorder=-1))
    text(.04,yf+.007,names[d],8.3,weight='bold',va='center');text(.04,yf-.012,f'{n:,} '+('drug ID' if n==1 else 'drug IDs'),7.4,GRAY,va='center')
    ax.plot([block.loc['logCPM1','pcc'],block.loc['logCPM100','pcc']],[y,y],color='#C2CDD5',lw=1.1,zorder=1)
    for mode,col,mark,xnum in [('logCPM1',TEAL,'o',.855),('logCPM100',AMBER,'D',.95)]:
        z=block.loc[mode];point(ax,z.pcc,y,z.low,z.high,col,mark,n>=5);text(xnum,yf,f'{z.pcc:.3f}',7.8,col,ha='center',va='center')
        records.append({'panel':'A','dataset':d,'normalization':mode,'estimate':float(z.pcc),'low':jvalue(z.low),'high':jvalue(z.high),'drugs':n,'axis':'common869','interval_drawn':bool(np.isfinite(z.low))})
text(.04,.568,'Gray lines join the same samples; a higher correlation does not establish greater fidelity.',7.4,GRAY)
fig.add_artist(Line2D([.04,.97],[.538,.538],transform=fig.transFigure,color=RULE,lw=.65))
panel(.503,'B','Repeatability and concordance on identical support')
text(.04,.471,'Matched cell line · 24 h · dose ratio ≤2 · primary 869-gene axis',7.7,GRAY)
bx=fig.add_axes([.34,.124,.63,.267]);bx.set_ylim(NPAIR-.5,-.5);style(bx,(-.10,.90),[0,.2,.4,.6,.8],'Single-repeat PCC')
fig.legend(handles=[Line2D([],[],color=c,marker=m,lw=1,ms=4,label=l) for c,m,l in [(BLUE,'s','First source'),(AMBER,'D','Second source'),(TEAL,'o','Across sources')]],loc='lower left',bbox_to_anchor=(.326,.414),frameon=False,ncol=3,fontsize=7.5,columnspacing=1.45,handlelength=1.3)
pairs=list(dict.fromkeys(zip(match.A,match.B)));assert len(pairs)==NPAIR
for y,(a,b) in enumerate(pairs):
    block=match[match.A.eq(a)&match.B.eq(b)].set_index('metric');assert len(block)==3 and block.drugs.nunique()==1 and block.bootstrap_units.nunique()==1
    n=int(block.drugs.iloc[0]);g=int(block.bootstrap_units.iloc[0]);yf=.124+.267*(NPAIR-.5-y)/NPAIR
    if y%2==0:fig.add_artist(Rectangle((.04,yf-.025),.93,.050,transform=fig.transFigure,fc=LIGHT,ec='none',zorder=-1))
    text(.04,yf+.008,names[a]+' / '+names[b],7.8,weight='bold',va='center');text(.04,yf-.013,f'{n} drug'+('s' if n!=1 else '')+f' · {g} resampling group'+('s' if g!=1 else ''),7.4,GRAY,va='center')
    for metric,dy,col,mark in [('rAA',-.21,BLUE,'s'),('rBB',0,AMBER,'D'),('rAB',.21,TEAL,'o')]:
        z=block.loc[metric];point(bx,z.estimate,y+dy,z.ci_low,z.ci_high,col,mark,n>=5)
        records.append({'panel':'B','first_source':a,'second_source':b,'metric':metric,'estimate':float(z.estimate),'low':jvalue(z.ci_low),'high':jvalue(z.ci_high),'drugs':n,'resampling_groups':g,'interval_drawn':bool(np.isfinite(z.ci_low)),'descriptive_small_n':n<5})
text(.04,.047,'Retained 95% intervals. Open markers: fewer than five drugs; descriptive only.',7.4,GRAY)
text(.04,.022,'MIX-seq has one repeat drug and no drug-level interval. Between-source gaps do not isolate platform effects.',7.4,GRAY)
fig.canvas.draw();renderer=fig.canvas.get_renderer();fw,fh=fig.canvas.get_width_height();texts=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()];minfont=min(t.get_fontsize() for t in texts)
assert minfont>=7.4
outside=[]
for t in texts:
    box=t.get_window_extent(renderer)
    if box.x0<-.5 or box.y0<-.5 or box.x1>fw+.5 or box.y1>fh+.5:outside.append(t.get_text())
assert not outside,outside
assert all(not re.search('[\u3400-\u9fff]',t.get_text()) for t in texts)
outputs={}
for ext in ['svg','png','pdf']:
    p=F/f'figS1_robustness.{ext}';fig.savefig(p,dpi=300,facecolor='white');outputs[str(p.relative_to(R))]=sha(p)
plt.close(fig)
assert all(sha(R/p)==h for p,h in hashes.items())
caption=('Figure S1. Robustness of response reproducibility and cross-source concordance. '
'A, Processing sensitivity on the primary 869-gene axis. All four count-based sources with retained paired log2(CPM + 1) and log2(CPM + 100) summaries are shown; pairs and drugs are identical between transforms within each source. Gray lines join the two estimates. Native-axis sensitivities remain in the source tables. The common axis was specified for the primary analysis, not selected from these results. '
'B, Within- and cross-source single-repeat correlations on exactly the same drug, cell, dose and time support, requiring confirmed identities, 24-hour treatment, doses within twofold and independent repeats on both sides. All eligible source pairs are shown; missing joint repeat supports are listed in the source tables. First and second source refer to the order in each row label. Lines show retained 95% bootstrap intervals. Open markers flag fewer than five drugs; Small matching sets are reported with their actual drug counts. MIX-seq supplies one repeat drug in A, so its drug-level interval is unavailable. Neither increased correlation after transformation nor a within/between-source gap identifies biological fidelity, a pure platform effect or a noise fraction.')
(R/'S1_CAPTION.txt').write_text(caption+'\n',encoding='utf-8')
qa = {'status': 'RENDERED', 'new_statistics': False, 'input_sha256': hashes, 'output_sha256': outputs, 'width_mm': 180, 'height_mm': 6.05 * 25.4, 'minimum_font_pt': minfont, 'text_outside_canvas': outside, 'all_common_axis_normalization_rows': len(norm), 'all_eligible_strict_repeatability_rows': len(match), 'records': records, 'selection_rule': {'A': 'All sources on the previously specified common869 primary axis with paired pseudocount estimates; not selected by result', 'B': 'All source pairs satisfying the pre-existing joint repeatability eligibility protocol', 'native_axis': 'Retained in source data, omitted from this concise display to avoid duplicating the primary/common contrast', 'small_n': 'All stored B intervals shown but hollow-marked and explicitly descriptive; no drug-level interval manufactured for one-drug MIX-seq'}, 'unchanged_inference': ['No attenuation correction', 'No causal platform attribution', 'No re-fitting or re-selection of statistical estimates']}
(R/'CONCISE_S1_QA.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'rendered':'figS1_robustness','height_mm':6.05*25.4,'records':len(records),'minfont_pt':minfont}))


