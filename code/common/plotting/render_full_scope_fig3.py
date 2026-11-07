"""Plot response-component shares on native and common gene axes in Figure 3."""
from pathlib import Path
import sys,os,json,hashlib,re
R=Path(__file__).resolve().parents[1];V=R.parent
os.environ['MPLCONFIGDIR']=str(R/'code/mplconfig_s12')
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,transforms
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
INK='#192D3A';TEAL='#147D92';GOLD='#D69B36';PURPLE='#7B66AD';BLUE='#7AA6C2';GRAY='#667D8C';LIGHT='#F3F6F8';RULE='#DCE4E9'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.2,'axes.titlesize':10.5,'axes.labelsize':8.2,'xtick.labelsize':8,'ytick.labelsize':8.2,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.edgecolor':RULE,'axes.linewidth':.65,'axes.unicode_minus':False,'svg.fonttype':'none','pdf.fonttype':42,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'savefig.facecolor':'white','hatch.linewidth':.45})
OUT=R/'figures';OUT.mkdir(exist_ok=True)
W=180/25.4;inputs={};records={};font_audit={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):inputs[str(p)]=sha(p);return pd.read_csv(p)
def err(ax,x,y,lo,hi,col=TEAL,marker='o',filled=True):
    if np.isfinite(lo) and np.isfinite(hi):ax.errorbar(x,y,xerr=[[max(0,x-lo)],[max(0,hi-x)]],fmt=marker,color=col,mfc=col if filled else 'white',ms=4.0,mew=.8,capsize=2,capthick=.75,lw=.9,zorder=4)
    else:ax.plot(x,y,marker,color=col,mfc=col if filled else 'white',ms=4.0,mew=.8,zorder=4)
def axis_style(ax,xlim,xlabel,xticks=None):
    ax.set_xlim(*xlim);ax.tick_params(axis='y',length=0);ax.tick_params(axis='x',length=3,pad=4);ax.set_yticks([]);ax.set_axisbelow(True);ax.grid(axis='x',color=RULE,lw=.55);ax.axvline(0,c='#9CAAB3',lw=.7,zorder=2);ax.set_xlabel(xlabel,labelpad=6)
    if xticks is not None:ax.set_xticks(xticks)
def table_text(ax,x,y,text,**kw):ax.text(x,y,text,transform=transforms.blended_transform_factory(ax.transAxes,ax.transData),**kw)
def header(fig,letter,title,y):
    fig.text(.035,y,letter,fontsize=12,weight='bold',va='baseline');fig.text(.075,y,title,fontsize=10.5,weight='bold',va='baseline')
def save(fig,stem,stats):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();texts=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()];sizes=[t.get_fontsize() for t in texts];assert min(sizes)>=7.2,(stem,min(sizes));font_audit[stem]={'width_mm':180,'minimum_pt':min(sizes)}
    fw,fh=fig.canvas.get_width_height();outside=[]
    for t in texts:
        b=t.get_window_extent(renderer)
        if b.x0<-.5 or b.y0<-.5 or b.x1>fw+.5 or b.y1>fh+.5:outside.append(t.get_text())
    font_audit[stem]['text_outside_canvas']=outside;assert not outside,(stem,outside)
    for ext in ['png','svg','pdf']:fig.savefig(OUT/f'{stem}.{ext}',dpi=300)
    records[stem]={'size_inches':list(fig.get_size_inches()),'plotted_records':stats};plt.close(fig)

# Figure 3: paired gene-axis bars, separated from source/support metadata.
full=read(R/'updated_inputs/figures/fig3a_full.csv');full=full[full.normalization.ne('logCPM100')]
names=['Tahoe','L1000_L3','CIGS','sciPlex3','MIXseq','DRUGseq'];display={'L1000_L3':'L1000 L3','MIXseq':'MIX-seq','DRUGseq':'DRUG-seq'}
comp=['mean','cell','drug','interaction_residual'];cols=[BLUE,TEAL,GOLD,PURPLE]
fig=plt.figure(figsize=(W,4.8));fig.text(.035,.953,'Observed response composition',weight='bold',fontsize=11)
for x,txt in [(.035,'SOURCE / SUPPORT'),(.278,'GENE AXIS'),(.447,'OBSERVED RESPONSE ENERGY')]:fig.text(x,.899,txt,fontsize=7.5,color='#627581',weight='bold')
ax=fig.add_axes([.447,.205,.523,.665]);ax.set_ylim(12.85,-.65);ax.set_xlim(0,1)
left=fig.add_axes([.035,.205,.382,.665]);left.set_ylim(ax.get_ylim());left.set_xlim(0,1);left.axis('off');stats=[]
for i,nm in enumerate(names):
    yc=i*2.35
    if i%2==0:
        ax.axhspan(yc-.45,yc+1.28,color=LIGHT,zorder=-2);left.axhspan(yc-.45,yc+1.28,color=LIGHT,zorder=-2)
    z0=full[(full.dataset==nm)&(full.axis=='native')].iloc[0]
    left.text(0,yc+.18,display.get(nm,nm),fontsize=9,weight='bold',va='center')
    support=f'{int(z0.drugs):,} IDs · '+('U2OS only' if nm=='DRUGseq' else f'{int(z0.cells)} cell lines')
    left.text(0,yc+.70,support,fontsize=7.4,color='#627581',va='center')
    for j,axis in enumerate(['native','common']):
        z=full[(full.dataset==nm)&(full.axis==axis)].iloc[0];y=yc+j*.83;start=0
        for c,col in zip(comp,cols):
            raw=float(z[c+'_fraction']);value=max(0,raw);mixed=nm=='DRUGseq' and c=='drug'
            ax.barh(y,value,left=start,height=.58,color='#D2D9DE' if mixed else col,edgecolor='#A6B3BC' if mixed else 'white',linewidth=.35,hatch='////' if mixed else None,zorder=3)
            if value>=.115:ax.text(start+value/2,y,f'{value*100:.1f}',ha='center',va='center',fontsize=7.3,color=INK if col in [GOLD,BLUE] or mixed else 'white',zorder=5)
            start+=value
        left.text(.636,y,('Native' if axis=='native' else 'Common')+f'  {int(z.genes):,}',fontsize=7.8,va='center')
        stats.append({'dataset':nm,'axis':axis,**{c:float(z[c+'_fraction']) for c in comp},'drug_IDs':int(z.drugs),'cell_lines':int(z.cells),'gene_entries':int(z.genes)})
ax.set_yticks([]);ax.set_xticks([0,.25,.5,.75,1],['0','25','50','75','100']);ax.tick_params(axis='x',length=3,pad=4);ax.set_xlabel('Share of squared response (%)',labelpad=6)
for a in [ax,left]:a.axhline(11.10,color='#AFBCC5',lw=.7,ls=(0,(3,2)),zorder=5)
fig.legend(handles=[Patch(color=BLUE,label='Overall (μ)'),Patch(color=TEAL,label='Cell (α)'),Patch(color=GOLD,label='Drug (β)'),Patch(color=PURPLE,label='Interaction residual (γ)')],loc='lower left',bbox_to_anchor=(.025,.080),ncol=4,frameon=False,fontsize=8,handlelength=1.4,columnspacing=1.8,handletextpad=.5)
fig.text(.035,.050,'Native counts: measured gene entries. Gray hatching: drug variation in one cell line.',fontsize=7.5,color='#627581')
fig.text(.035,.024,'Residuals include measurement error; DRUG-seq cannot separate β from γ.',fontsize=7.5,color='#627581')
save(fig,'fig3a_full',stats)

