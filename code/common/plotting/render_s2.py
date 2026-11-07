from pathlib import Path
import os,csv,io,html
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
ROOT=Path(os.environ['TAHOE_FIGURE_WORKDIR']);R=ROOT/'supplement'
for d in ['figures','output/pdf']:(R/d).mkdir(parents=True,exist_ok=True)
rows=list(csv.DictReader((ROOT/'base/inputs/EXPLORATORY_CONFIGURATIONS.csv').open(encoding='utf-8-sig',newline='')))
assert len(rows)==10 and all(r['delta'] for r in rows)
import matplotlib
_fontdir=Path(matplotlib.get_data_path())/'fonts/ttf'
pdfmetrics.registerFont(TTFont('DejaVu Sans',str(_fontdir/'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DejaVu Sans Bold',str(_fontdir/'DejaVuSans-Bold.ttf')))
W=174/25.4*72;H=493
C=canvas.Canvas(str(R/'output/pdf/Figure_S2.pdf'),pagesize=(W,H))
C.setTitle('Figure S2 | Predicting drug-specific main effects')
C.setAuthor('Jinming Bai; Sharon Prince; Geoff S. Nitschke')
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="174mm" height="{H*25.4/72:.3f}mm" viewBox="0 0 {W} {H}">',f'<rect width="{W}" height="{H}" fill="white"/>']
text_checks=[]
ink='#20343F'; muted='#607C8D'; grid='#D8E2E8'
colors={'structure':'#147F97','mechanism':'#8265AA','measurement':'#B68120'}
def compact_y(y):return y-18 if y>=50 else y
def line(x1,y1,x2,y2,color=grid,width=.5,dash=False):
    y1,y2=compact_y(y1),compact_y(y2)
    C.setStrokeColor(HexColor(color));C.setLineWidth(width);C.setDash([2,2] if dash else [])
    C.line(x1,H-y1,x2,H-y2)
    dd=' stroke-dasharray="2 2"' if dash else ''
    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{dd}/>')
def rect(x,y,w,h,color):
    y=compact_y(y)
    C.setFillColor(HexColor(color));C.rect(x,H-y-h,w,h,fill=1,stroke=0)
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{color}"/>')
def dot(x,y,color,r=2.4):
    y=compact_y(y)
    C.setFillColor(HexColor(color));C.circle(x,H-y,r,fill=1,stroke=0)
    svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}"/>')
def text(x,y,s,size=8.4,color=ink,bold=False,anchor='start'):
    y=compact_y(y)
    font='DejaVu Sans Bold' if bold else 'DejaVu Sans';width=pdfmetrics.stringWidth(s,font,size)
    xx=x-(width/2 if anchor=='middle' else width if anchor=='end' else 0)
    assert xx>=0 and xx+width<=W,(s,xx,width)
    C.setFont(font,size);C.setFillColor(HexColor(color));C.drawString(xx,H-y,s)
    svg.append(f'<text x="{x}" y="{y}" fill="{color}" font-family="DejaVu Sans" font-size="{size}" font-weight="{700 if bold else 400}" text-anchor="{anchor}">{html.escape(s)}</text>')
    text_checks.append({'text':s,'x0':xx,'x1':xx+width,'baseline':y,'font_pt':size})

text(16,24,'Predicting drug-specific main effects',12,bold=True)
for x,name,key in [(18,'Structure','structure'),(125,'Target annotation','mechanism'),(275,'External same-drug measurements','measurement')]:
    dot(x,61,colors[key],2.7);text(x+8,64,name,8.2)
text(16,88,'Input',8.5,bold=True);text(144,88,'Model (K)',8.5,bold=True)
text(241,88,'Drugs',8.5,bold=True,anchor='middle')
text(316,88,'Uncovered β PCC',8.5,bold=True,anchor='middle')
text(433,88,'ΔPCC vs permutation',8.5,bold=True,anchor='middle')
line(16,97,W-16,97)
text(16,115,'Chemical-group holdouts',9.4,bold=True)
text(16,370,'Chemical-group and known-MoA holdouts',9.4,bold=True)
line(16,351,W-16,351)
ys=[140+30*j for j in range(7)]+[397+30*j for j in range(3)]
labels=[('KPGT structure','Ridge (3)'),('KPGT structure','Similarity (10)'),('Predicted targets','Similarity (3)'),('Annotated targets','Program kNN (10)'),('ChEMBL affinities','CCA (10)'),('CIGS response','Ridge (3)'),('DRUG-seq response','Ridge (3)'),('Predicted targets','Ridge (10)'),('Annotated targets','Ridge (10)'),('ChEMBL affinities','Ridge (10)')]
xa=(272,360);xb=(392,480);xmin=-.23;xmax=.20
def X(v,axis):return axis[0]+(v-xmin)/(xmax-xmin)*(axis[1]-axis[0])
for idx,y in enumerate(ys):
    if idx%2:rect(16,y-12,W-32,27,'#F3F6F8')
for axis in [xa,xb]:
    for tick in [-.2,0,.2]:
        x=X(tick,axis)
        line(x,125,x,334,muted if tick==0 else grid,.55,tick==0)
        line(x,382,x,470,muted if tick==0 else grid,.55,tick==0)
        text(x,492,f'{tick:.1f}',8,anchor='middle')
    line(axis[0],479,axis[1],479)
for r,y,(label,model) in zip(rows,ys,labels):
    text(16,y+3,label,8.4);text(144,y+3,model,8.1);text(241,y+3,r['n'],8.4,anchor='middle')
    color=colors[r['family']]
    for axis,cols in [(xa,('value','lo','hi')),(xb,('delta','dlo','dhi'))]:
        if not r[cols[0]]:
            text((axis[0]+axis[1])/2,y+3,'Not available',7.8,color=muted,anchor='middle');continue
        v,lo,hi=[float(r[k]) for k in cols];assert xmin<lo<=v<=hi<xmax
        xx,xlo,xhi=[X(vv,axis) for vv in [v,lo,hi]]
        line(xlo,y,xhi,y,color,1);line(xlo,y-2,xlo,y+2,color,.7);line(xhi,y-2,xhi,y+2,color,.7);dot(xx,y,color)
        text(xx,y-5,f'{v:+.3f}',7.8,color=color,anchor='middle')
C.showPage();C.save();svg.append('</svg>')
(R/'figures/Figure_S2.svg').write_text('\n'.join(svg),encoding='utf-8')
import pymupdf as fitz
doc=fitz.open(R/'output/pdf/Figure_S2.pdf')
doc[0].get_pixmap(matrix=fitz.Matrix(3,3),alpha=False).save(R/'figures/Figure_S2.png')
