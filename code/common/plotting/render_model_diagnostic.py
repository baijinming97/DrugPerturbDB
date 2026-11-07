"""Plot baseline component scores and measured-input prediction diagnostics."""
from pathlib import Path
import os,sys,json,hashlib,re
sys.dont_write_bytecode=True
O=Path(__file__).resolve().parents[1];V=O.parent;F=O/'figures';F.mkdir(exist_ok=True)
P=V/'PREPRINT_FULL_DATA_V1';S=V/'SUPPLEMENT_ONLY_REVIEW_V4_20261102'
os.environ['MPLCONFIGDIR']=str(O/'code/mplconfig_fig4_s34')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch,Rectangle,FancyArrowPatch
from matplotlib.text import Text
from matplotlib import font_manager
INK='#192D3A';TEAL='#147D92';GOLD='#D69B36';PURPLE='#7B66AD';BLUE='#7AA6C2';GRAY='#667D8C';LIGHT='#F3F6F8';LINE='#DCE4E9'
cols=[BLUE,TEAL,GOLD,PURPLE];WIDTH=180/25.4
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'savefig.facecolor':'white','axes.edgecolor':'#B5C3CC','axes.linewidth':.55,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'xtick.labelsize':7.5,'ytick.labelsize':8,'lines.solid_capstyle':'round'})
inputs={};out=[];size_audit={};plotted={};active='';connections=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):inputs[str(p)]=sha(p);return pd.read_csv(p,float_precision='round_trip')
def new(stem,height):
 global active;active=stem;plotted[stem]=[]
 return plt.figure(figsize=(WIDTH,height),facecolor='white')
def text(fig,x,y,s,size=8,color=INK,weight='normal',**kw):return fig.text(x,y,s,fontsize=size,color=color,weight=weight,**kw)
def title(fig,s,sub):
 text(fig,.04,.97,s,10.7,weight='bold',va='top');text(fig,.04,.925,sub,7.6,GRAY,va='top')
def panel(fig,x,y,letter,s):
 text(fig,x,y,letter,12,weight='bold',va='center');text(fig,x+.036,y,s,9.2,weight='bold',va='center')
def rule(fig,y):fig.add_artist(Line2D([.04,.97],[y,y],transform=fig.transFigure,color=LINE,lw=.65))
def point(ax,x,y,lo,hi,color,marker='o',filled=True,size=4,ci=True,tag=''):
 plotted[active].append({'tag':tag,'value':float(x),'low':float(lo),'high':float(hi),'interval_shown':bool(ci)})
 if ci:
  ax.plot([lo,hi],[y,y],color=color,lw=1.15,zorder=3)
  ax.plot([lo,hi],[y,y],ls='none',marker='|',ms=4,color=color,mew=.7,zorder=3)
 ax.plot(x,y,marker=marker,ms=size,mfc=color if filled else 'white',mec=color,mew=.9,ls='none',zorder=4)
def prep(ax,zero=True):
 if zero:ax.axvline(0,color='#A2AFB8',lw=.7,ls=(0,(3,3)),zorder=1)
 ax.grid(axis='x',color=LINE,lw=.45);ax.set_axisbelow(True);ax.tick_params(axis='y',length=0,pad=5);ax.tick_params(axis='x',length=3,width=.55);ax.spines['left'].set_visible(False)
def box(ax,x,y,w,h,label,fc=LIGHT,fontsize=8):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=0.009',transform=ax.transAxes,fc=fc,ec=LINE,lw=.65,clip_on=False))
 ax.text(x+w/2,y+h/2,label,transform=ax.transAxes,ha='center',va='center',fontsize=fontsize)
 return {'x':x,'y':y,'w':w,'h':h,'label':label}

def arrow(ax,start,end,label,orientation,margin_mm):
 # Endpoints are exact axes coordinates, with no annotation's hidden shrink.
 # Filled arrowheads and a visible shaft remain legible at publication width.
 ax.add_patch(FancyArrowPatch(start,end,transform=ax.transAxes,arrowstyle='-|>',
                             color=GRAY,lw=.8,mutation_scale=7.0,
                             shrinkA=0,shrinkB=0,clip_on=False))
 pos=ax.get_position();mm=np.array([pos.width*ax.figure.get_figwidth(),pos.height*ax.figure.get_figheight()])*25.4
 delta=(np.array(end)-np.array(start))*mm
 assert abs(delta[1 if orientation=='horizontal' else 0])<1e-12
 assert np.linalg.norm(delta)>1.8
 connections.append({'label':label,'orientation':orientation,'start':list(start),'end':list(end),
                     'axis_alignment_error_mm':float(abs(delta[1 if orientation=='horizontal' else 0])),
                     'length_mm':float(np.linalg.norm(delta)),'endpoint_clearance_mm':margin_mm})

def connect_boxes(ax,left,right,orientation='horizontal',margin_mm=.65):
 pos=ax.get_position();mm=np.array([pos.width*ax.figure.get_figwidth(),pos.height*ax.figure.get_figheight()])*25.4
 if orientation=='horizontal':
  cy=left['y']+left['h']/2;assert abs(cy-right['y']-right['h']/2)<1e-12
  start=(left['x']+left['w']+margin_mm/mm[0],cy);end=(right['x']-margin_mm/mm[0],cy)
 else:
  cx=left['x']+left['w']/2;assert abs(cx-right['x']-right['w']/2)<1e-12
  start=(cx,left['y']-margin_mm/mm[1]);end=(cx,right['y']+right['h']+margin_mm/mm[1])
 arrow(ax,start,end,left['label'].replace('\n',' ')+' → '+right['label'].replace('\n',' '),orientation,margin_mm)

def connect_texts(ax,left,right,margin_mm=1.2):
 ax.figure.canvas.draw();renderer=ax.figure.canvas.get_renderer();inv=ax.transAxes.inverted()
 a=inv.transform_bbox(left.get_window_extent(renderer));b=inv.transform_bbox(right.get_window_extent(renderer))
 cy=(a.y0+a.y1)/2;assert abs(cy-(b.y0+b.y1)/2)<1e-10
 mm=ax.get_position().width*ax.figure.get_figwidth()*25.4
 arrow(ax,(a.x1+margin_mm/mm,cy),(b.x0-margin_mm/mm,cy),left.get_text()+' → '+right.get_text(),'horizontal',margin_mm)
def save(fig,stem):
 fig.canvas.draw();renderer=fig.canvas.get_renderer();fw,fh=fig.canvas.get_width_height()
 texts=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
 assert all(not re.search('[\u3400-\u9fff]',t.get_text()) for t in texts)
 minfont=min(t.get_fontsize() for t in texts);assert minfont>=7.3,(stem,minfont)
 outside=[]
 for t in texts:
  b=t.get_window_extent(renderer)
  if b.x0 < -1 or b.y0 < -1 or b.x1 > fw+1 or b.y1 > fh+1:outside.append(t.get_text())
 assert not outside,(stem,outside)
 size_audit[stem]={'width_mm':180,'height_mm':float(fig.get_figheight()*25.4),'minimum_font_pt':minfont,'text_outside_canvas':outside}
 for ext in ['svg','png']:
  p=F/f'{stem}.{ext}';fig.savefig(p,dpi=300,facecolor='white');out.append(str(p))
 plt.close(fig)

# A-D: protocol strip, aligned evidence table, diagnostic forest.
tab=read(P/'evidence_model/FOUR_COMPONENT_COMPARISON.csv');b=tab[tab.model.eq('B')].set_index('component')
stored=read(P/'evidence_model/BASELINE_STORED.csv').iloc[0]
ot=read(P/'evidence_model/PRIMARY_ORACLE_DIAGNOSTIC.csv');o=ot[ot.workflow.eq('WS1')&ot.setting.eq('oracle_beta_decoder')].iloc[0]
parts=['mu','alpha','beta','gamma'];errpct=b.loc[parts,'MSE'].to_numpy()/float(b.loc['FULL_DELTA','MSE'])*100
assert abs(b.loc[parts,'MSE'].sum()-b.loc['FULL_DELTA','MSE'])<1e-14
fig=new('fig4_focus',6.25)
title(fig,'X1-only coverage and information in measured drug effects','Development panel  ·  153 drugs  ·  47 cell lines  ·  978 genes  ·  5 μM')
panel(fig,.04,.865,'A','X1-only prediction and decomposition')
ax=fig.add_axes([.04,.690,.93,.146]);ax.axis('off')
nodes=[box(ax,0,.56,.18,.40,'Control expression\nX1 · 978 genes')]
gap=(1-(.18+3*.11+.225))/4
x=.18+gap
for j in range(3):
 nodes.append(box(ax,x,.56,.11,.40,f'Hidden {j+1}\n2,048'));x+=.11+gap
nodes.append(box(ax,x,.56,.225,.40,'Complete response\nΔX · 978 genes','#EDE8F5'))
for a,z in zip(nodes[:-1],nodes[1:]):connect_boxes(ax,a,z)
assert max(nodes[j+1]['x']-nodes[j]['x']-nodes[j]['w'] for j in range(4))-min(nodes[j+1]['x']-nodes[j]['x']-nodes[j]['w'] for j in range(4))<1e-12
ax.text(.465,.36,'Fully connected + ReLU + dropout (0.1)',ha='center',va='center',fontsize=7.5,color=GRAY)
labels=[ax.text(.155,.08,'Predicted and measured ΔX',ha='center',va='center',fontsize=8),
        ax.text(.510,.08,'Weighted decomposition',ha='center',va='center',fontsize=8),
        ax.text(.875,.08,'μ  +  α  +  β  +  γ',ha='center',va='center',fontsize=9,weight='bold')]
for a,z in zip(labels[:-1],labels[1:]):connect_texts(ax,a,z)
text(fig,.04,.665,'No drug input; components are evaluated after one complete prediction.',7.6,GRAY);rule(fig,.640)
panel(fig,.04,.610,'B','Component prediction');panel(fig,.585,.610,'C','Error allocation')
ax=fig.add_axes([.04,.386,.485,.196]);ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
for x,s,h in [(0,'Component','left'),(.70,'PCC','right'),(.995,'MSE','right')]:ax.text(x,.96,s,fontsize=7.6,color=GRAY,ha=h)
names=['Overall response μ','Cell effect α','Drug effect β','Interaction residual γ']
for i,(c,nm,col) in enumerate(zip(parts,names,cols)):
 y=.76-i*.205;ax.plot([0,1],[y-.105,y-.105],color=LINE,lw=.5)
 ax.add_patch(Rectangle((0,y-.035),.012,.07,color=col,transform=ax.transAxes));ax.text(.03,y,nm,va='center',fontsize=8)
 ax.text(.70,y,f'{b.loc[c,"PCC"]:.3f}',ha='right',va='center',fontsize=9,weight='bold');ax.text(.995,y,f'{b.loc[c,"MSE"]:.6f}',ha='right',va='center',fontsize=8)
 plotted[active].append({'component':c,'PCC':float(b.loc[c,'PCC']),'MSE':float(b.loc[c,'MSE']),'error_pct':float(errpct[i])})
ax=fig.add_axes([.635,.428,.325,.146]);ax.barh(range(4),errpct,color=cols,height=.52)
for i,v in enumerate(errpct):ax.text(v+1.0,i,f'{v:.2f}%',va='center',fontsize=8,weight='bold')
ax.set_yticks(range(4),['μ','α','β','γ']);ax.set_ylim(3.55,-.55);ax.set_xlim(0,63);ax.set_xticks([0,20,40,60]);ax.set_xlabel('Share of complete MSE (%)',fontsize=7.6,labelpad=3);prep(ax,False)
text(fig,.04,.351,'Complete ΔX   PCC 0.5821   |   MSE 0.007005',8.4,weight='bold');text(fig,.585,.340,'β + γ: 95.9%  |  Uncovered: 78.2%',7.6,GRAY);rule(fig,.327)
panel(fig,.04,.300,'D','Measured β input: development-panel diagnostic')
ax=fig.add_axes([.04,.100,.245,.185]);ax.axis('off')
nodes=[box(ax,0,y,.98,.20,lab,fc,7.6) for y,lab,fc in [(.77,'Measured uncovered β','#FBF2E1'),(.42,'β-to-γ Ridge decoder',LIGHT),(.07,'Projection + calibration','#EEEAF5')]]
for a,z in zip(nodes[:-1],nodes[1:]):connect_boxes(ax,a,z,'vertical',.35)
ax=fig.add_axes([.44,.117,.285,.126])
for i,key in enumerate(['gamma_PCC','full_PCC']):
 for off,src,color in [(-.16,stored,GRAY),(.16,o,PURPLE)]:
  val=float(src[key]);lo=float(src[key+'_lo']);hi=float(src[key+'_hi']);point(ax,val,i+off,lo,hi,color,size=4,tag=key+(' baseline' if off<0 else ' corrected'))
  ax.text(val,i+off+(-.17 if off<0 else .17),f'{val:.3f}',ha='center',va='bottom' if off<0 else 'top',color=color,fontsize=7.5)
ax.set_yticks(range(2),['Overall γ','Complete ΔX']);ax.set_ylim(1.9,-.65);ax.set_xlim(.40,.79);ax.set_xticks([.4,.5,.6,.7]);ax.set_xlabel('PCC · 95% interval',fontsize=7.6,labelpad=3);prep(ax,False)
fig.legend(handles=[Line2D([],[],marker='o',color=GRAY,lw=1,ms=4,label='X1-only'),Line2D([],[],marker='o',color=PURPLE,lw=1,ms=4,label='+ inferred γ')],loc='upper left',bbox_to_anchor=(.42,.282),frameon=False,ncol=2,fontsize=7.5,columnspacing=1.0,handlelength=1.3,handletextpad=.5)
axm=fig.add_axes([.78,.117,.19,.126]);axm.axis('off');axm.set_ylim(1.9,-.65);axm.text(.5,1.23,'MSE',transform=axm.transAxes,ha='center',fontsize=8,weight='bold')
for i,key in enumerate(['gamma_MSE','full_MSE']):
 axm.text(.5,i-.16,f'{float(stored[key]):.6f}',ha='center',va='center',fontsize=7.8,color=GRAY);axm.text(.5,i+.16,f'{float(o[key]):.6f}',ha='center',va='center',fontsize=7.8,color=PURPLE)
 plotted[active].append({'endpoint':key,'baseline':float(stored[key]),'corrected':float(o[key])})
text(fig,.04,.067,'Uncovered γ: r = 0.664 [0.625, 0.698]',8.2,PURPLE,weight='bold');text(fig,.04,.026,'Measured-input diagnostic; only inferred γ is added. Error includes measurement noise.',7.5,GRAY)
save(fig,'fig4_focus')

# E: each permutation and both transfer directions remain visible.
e2=read(V/'PAPER_TWO_CHECKS_V1/results/exp2/WITHIN_CROSS_PLATE.csv');lookup={(r.model,r.direction):r for r in e2.itertuples(index=False)}
assert set(e2.weighting)=={'common_support_group_equal'}
directions=['6_to_6','14_to_14','6_to_14','14_to_6'];models=['G5_true','shuffle_20261103','shuffle_20261104','shuffle_20261105']
fig=new('FIG4E_CROSS_PLATE_LINK',3.8);panel(fig,.04,.95,'E','Measured β predicts interactions across plates')
text(fig,.04,.888,'Source-plate β → fixed decoder → target-plate γ  |  47 drugs · 47 cells · 978 genes',7.6,GRAY)
ax=fig.add_axes([.245,.18,.72,.51])
for y,d in enumerate(directions):
 if y>=2:ax.axhspan(y-.5,y+.5,color='#EDF5F6',zorder=0)
 for j,m in enumerate(models):
  r=lookup[m,d];assert r.drugs==47 and r.chemical_groups==36
  point(ax,r.gamma_PCC,y+[-.24,-.07,.10,.27][j],r.gamma_PCC_lo,r.gamma_PCC_hi,TEAL if j==0 else GRAY,marker=['o','s','^','D'][j],filled=j==0,size=4.4 if j==0 else 3.3,tag=m+' '+d)
ax.set_yticks(range(4),['Plate 6 → 6','Plate 14 → 14','Plate 6 → 14','Plate 14 → 6']);ax.set_ylim(3.55,-.55);ax.set_xlim(-.24,.86);ax.set_xticks([-.2,0,.2,.4,.6,.8]);ax.set_xlabel('Target-plate interaction PCC',labelpad=4);prep(ax);ax.axhline(1.5,color='#CAD7DE',lw=.6)
fig.legend(handles=[Line2D([],[],color=TEAL,marker='o',ms=4,label='Measured β')]+[Line2D([],[],color=GRAY,marker=m,mfc='white',ms=3.5,label=f'Permutation {i}') for i,m in enumerate(['s','^','D'],1)],ncol=4,frameon=False,loc='upper left',bbox_to_anchor=(.235,.82),columnspacing=1.05,fontsize=7.5,handlelength=1.4,handletextpad=.45)
text(fig,.04,.075,'Shading: cross-plate predictions. 95% chemical-group intervals.',7.5,GRAY);text(fig,.04,.030,'β uses all source-plate cells. Complete-response outcomes: Figure S2C.',7.5,GRAY)
save(fig,'FIG4E_CROSS_PLATE_LINK')

