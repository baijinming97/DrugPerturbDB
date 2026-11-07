"""Lay out the measured-input diagnostic and prediction scores in Figure 4D."""
from pathlib import Path
import ast, csv, hashlib, html, io, json, re, shutil, sys, tarfile
import xml.etree.ElementTree as ET
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import os
ROOT=Path(os.environ['TAHOE_FIGURE_WORKDIR'])
R=ROOT/'final_main';V=ROOT
S=ROOT/'schematic'
P=ROOT/'base/evidence_inputs/PREPRINT_FULL_DATA_V1/evidence_model'
import pymupdf as fitz

for name in ['figures','output/pdf','qa','data']:(R/name).mkdir(parents=True,exist_ok=True)
F=R/'figures';O=R/'output/pdf';Q=R/'qa'
NS='http://www.w3.org/2000/svg';XL='http://www.w3.org/1999/xlink'
ET.register_namespace('',NS);ET.register_namespace('xlink',XL)
tag=lambda s:'{'+NS+'}'+s
W=510.236220472441;SCALE=174/180
INK='#192d3a';LINE='#667d8c';BORDER='#d7e2ea';FILL='#f3f6f8';PURPLE='#7b66ad'
import matplotlib
_fontdir=Path(matplotlib.get_data_path())/'fonts/ttf'
pdfmetrics.registerFont(TTFont('DejaVu Sans',str(_fontdir/'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DejaVu Sans Bold',str(_fontdir/'DejaVuSans-Bold.ttf')))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

exec(compile((Path(__file__).parent/'vector_primitives.py').read_text(encoding='utf-8'),'vector_primitives.py','exec'))

class Panel(Drawing):
    def line(self,x1,y1,x2,y2,color=BORDER,lw=.5):
        ET.SubElement(self.group,tag('line'),{'x1':str(x1),'y1':str(y1),'x2':str(x2),'y2':str(y2),'stroke':color,'stroke-width':str(lw)})
        self.color(color,True);self.pdf.setLineWidth(lw)
        self.pdf.line(x1,self.height-y1,x2,self.height-y2)
    def circle(self,x,y,r,color):
        ET.SubElement(self.group,tag('circle'),{'cx':str(x),'cy':str(y),'r':str(r),'fill':color})
        self.color(color);self.pdf.circle(x,self.height-y,r,stroke=0,fill=1)
    def point(self,value,lo,hi,y,color):
        x=lambda z:250+(z-.4)/.39*133
        self.line(x(lo),y,x(hi),y,color,1.0)
        self.line(x(lo),y-2,x(lo),y+2,color,.65)
        self.line(x(hi),y-2,x(hi),y+2,color,.65)
        self.circle(x(value),y,2.5,color)

expected={
 'BASELINE_STORED.csv':'4d444ffff5067252c7b18d0244765fcd1f76dc735303ccb89a6356ce627cba0d',
 'PRIMARY_ORACLE_DIAGNOSTIC.csv':'1dfbc5fef2e7d46b2f6e5e22602e3ca99336b9cc514ed3e7f9caf1b508f4b5c7'}
for name,digest in expected.items():assert sha(P/name)==digest,(name,sha(P/name))
def rows(name):
    with (P/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
b=rows('BASELINE_STORED.csv')[0]
o=next(r for r in rows('PRIMARY_ORACLE_DIAGNOSTIC.csv') if r['workflow']=='WS1' and r['setting']=='oracle_beta_decoder')
records=[]
for target,prefix in [('Overall γ','gamma'),('Complete ΔX','full')]:
    for model,row in [('Control-only',b),('Control-only + inferred γ',o)]:
        records.append(dict(target=target,model=model,PCC=float(row[prefix+'_PCC']),PCC_lo=float(row[prefix+'_PCC_lo']),PCC_hi=float(row[prefix+'_PCC_hi']),MSE=float(row[prefix+'_MSE'])))
with (R/'data/Figure_4D.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

# The left side is the measured-input diagnostic; the right side contains the
# assembled prediction metrics. They have distinct targets and labels.
TOP=277;OLD_END=397;DH=134;SHIFT=DH-(OLD_END-TOP)
d=Panel(DH);d.group.set('id','panel_D_refined')
d.text(20.409449,10.5,'D',12,True,anchor='start')
d.text(38.777953,9.8,'Interaction prediction from measured drug effects',9.2,True,anchor='start')
d.rect(20.409449,34,124,24,fill='#fbf2e1',r=1.2)
d.text(82.409449,49,'Measured uncovered β',7.8)
d.arrow([(82.409449,59.3),(82.409449,66.5)])
d.rect(20.409449,68,124,24,r=1.2)
d.text(82.409449,83,'β-to-γ Ridge decoder',7.8)
d.arrow([(82.409449,93.3),(82.409449,100.5)])
d.rect(20.409449,102,124,24,fill='#eeebf5',r=1.2)
d.text(82.409449,117,'Inferred γ correction',8.0,True)

# A single legend applies to PCC, its interval, and the aligned MSE entry.
for x,color,label in [(247,LINE,'Control-only'),(332,PURPLE,'Control-only + inferred γ')]:
    d.line(x-5,29,x+5,29,color,.9);d.circle(x,29,2.1,color)
    d.text(x+9,31.5,label,7.5,color=INK,anchor='start')
d.text(164,47,'Prediction target',7.8,color=LINE,anchor='start')
d.text(330,47,'PCC',8,True)
d.text(470,47,'MSE',8,True)
d.line(164,51,490.4,51)
for z in [.4,.5,.6,.7]:
    x=250+(z-.4)/.39*133
    d.line(x,54,x,119,'#e7edf1',.45)
    d.line(x,121,x,124,LINE,.45)
    d.text(x,132,f'{z:.1f}',7.5)
d.line(250,121,383,121,color='#b5c3cc',lw=.55)
d.line(164,86,490.4,86)
d.text(164,71,'Overall γ',8,anchor='start')
d.text(164,107,'Complete ΔX',8,anchor='start')
for r,y,color in zip(records,[61,76,97,112],[LINE,PURPLE,LINE,PURPLE]):
    d.point(r['PCC'],r['PCC_lo'],r['PCC_hi'],y,color)
    d.text(419,y+2.7,f'{r["PCC"]:.3f}',8.2,bold=True,color=color,anchor='end')
    d.text(470,y+2.7,f'{r["MSE"]:.6f}',8,color=color)
panel=d.finish()

# Compose panel D with the vector content of panels A-C and E.
source=fitz.open(S/'output/pdf/Figure_4.pdf');sp=source[0]
out=fitz.open();height=638+SHIFT
np=out.new_page(width=W*SCALE,height=height*SCALE)
np.show_pdf_page(fitz.Rect(0,0,W*SCALE,TOP*SCALE),source,0,clip=fitz.Rect(0,0,W*SCALE,TOP*SCALE))
np.show_pdf_page(fitz.Rect(0,TOP*SCALE,W*SCALE,(TOP+DH)*SCALE),panel,0)
np.show_pdf_page(fitz.Rect(0,(OLD_END+SHIFT)*SCALE,W*SCALE,height*SCALE),source,0,clip=fitz.Rect(0,OLD_END*SCALE,W*SCALE,sp.rect.height))
out.subset_fonts();out.save(O/'Figure_4.pdf',deflate=True,garbage=4)
np.get_pixmap(dpi=300).save(F/'Figure_4.png')
panel[0].get_pixmap(dpi=300).save(F/'Figure_4D.png')

root=ET.parse(S/'figures/Figure_4.svg').getroot()
root.set('viewBox',f'0 0 {W} {height}');root.set('height',f'{height*SCALE/72*25.4:.5f}mm')
for child in root:
    if child.tag==tag('rect'):child.set('height',str(height))
    if child.tag==tag('g') and child.get('transform','').startswith('translate(0 '):
        yy=float(re.search(r'translate\(0 ([\d.]+)\)',child.get('transform')).group(1))
        if yy>=409:child.set('transform',f'translate(0 {yy+SHIFT})')
for cl in root.iter(tag('clipPath')):
    if cl.get('id')=='band_4_1':next(iter(cl)).set('height',str(TOP-128))
g=ET.SubElement(root,tag('g'),{'transform':f'translate(0 {TOP})'})
g.append(d.group)
ET.ElementTree(root).write(F/'Figure_4.svg',encoding='utf-8',xml_declaration=True)

preservation=[]
for label,oldtop,oldbottom,newtop in [('A-C',0,TOP,0),('E',OLD_END,638,OLD_END+SHIFT)]:
    h=(oldbottom-oldtop)*SCALE
    a=fitz.open();ap=a.new_page(width=W*SCALE,height=h)
    ap.show_pdf_page(ap.rect,source,0,clip=fitz.Rect(0,oldtop*SCALE,W*SCALE,oldbottom*SCALE))
    z=fitz.open();zp=z.new_page(width=W*SCALE,height=h)
    zp.show_pdf_page(zp.rect,out,0,clip=fitz.Rect(0,newtop*SCALE,W*SCALE,newtop*SCALE+h))
    assert ap.get_text()==zp.get_text(),label
    assert ap.get_pixmap(dpi=144).samples==zp.get_pixmap(dpi=144).samples,label
    preservation.append({'panels':label,'text_and_pixels_identical':True})
    a.close();z.close()

spans=[s for bl in np.get_text('dict')['blocks'] for l in bl.get('lines',[]) for s in l['spans']]
assert min(s['size'] for s in spans)>=7
assert all('DejaVuSans' in s['font'] for s in spans)
assert not [s for s in spans if not (np.rect+(-.4,-.4,.4,.4)).contains(fitz.Rect(s['bbox']))]
numerictext=panel[0].get_text()
for r in records:
    assert f'{r["PCC"]:.3f}' in numerictext
    assert f'{r["MSE"]:.6f}' in numerictext
out.close();panel.close();source.close()

for i in [1,2,3]:
    for ext in ['png','svg']:shutil.copy2(S/f'figures/Figure_{i}.{ext}',F/f'Figure_{i}.{ext}')
    shutil.copy2(S/f'output/pdf/Figure_{i}.pdf',O/f'Figure_{i}.pdf')
