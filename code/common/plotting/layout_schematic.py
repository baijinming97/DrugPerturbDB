"""Position the Figure 3 legend and draw the Figure 4A prediction schematic."""
from pathlib import Path
import copy, hashlib, io, json, re, shutil, sys
import xml.etree.ElementTree as ET
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import os
ROOT=Path(os.environ['TAHOE_FIGURE_WORKDIR'])
R=ROOT/'schematic'
V=R.parent
S=ROOT/'layout'
import pymupdf as fitz
from PIL import Image, ImageChops

for d in ['figures','output/pdf','qa']: (R/d).mkdir(parents=True,exist_ok=True)
F=R/'figures'; O=R/'output/pdf'; Q=R/'qa'
NS='http://www.w3.org/2000/svg'; XL='http://www.w3.org/1999/xlink'
ET.register_namespace('',NS);ET.register_namespace('xlink',XL)
tag=lambda s:'{'+NS+'}'+s
W=510.236220472441; SCALE=174/180
import matplotlib
_fontdir=Path(matplotlib.get_data_path())/'fonts/ttf'
pdfmetrics.registerFont(TTFont('DejaVu Sans',str(_fontdir/'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DejaVu Sans Bold',str(_fontdir/'DejaVuSans-Bold.ttf')))
INK='#192d3a'; LINE='#667d8c'; BORDER='#d7e2ea'; FILL='#f3f6f8'
COLORS=['#7aa6c2','#137e92','#d99c2c','#7b66ad']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

class Drawing:
    """Emit identical vector primitives into an SVG group and a PDF."""
    def __init__(self,height):
        self.height=height;self.group=ET.Element(tag('g'),{'id':'refined_schematic'})
        self.buffer=io.BytesIO()
        self.pdf=canvas.Canvas(self.buffer,pagesize=(W*SCALE,height*SCALE),pageCompression=1)
        self.pdf.scale(SCALE,SCALE)
    def color(self,color,stroke=False):
        rgb=[int(color[i:i+2],16)/255 for i in [1,3,5]]
        (self.pdf.setStrokeColorRGB if stroke else self.pdf.setFillColorRGB)(*rgb)
    def rect(self,x,y,w,h,fill=FILL,stroke=BORDER,r=0,lw=.6):
        a={'x':str(x),'y':str(y),'width':str(w),'height':str(h),'fill':fill,'stroke':stroke,'stroke-width':str(lw)}
        if r:a.update(rx=str(r),ry=str(r))
        ET.SubElement(self.group,tag('rect'),a)
        self.color(fill);self.color(stroke,True);self.pdf.setLineWidth(lw)
        if r:self.pdf.roundRect(x,self.height-y-h,w,h,r,fill=1,stroke=1)
        else:self.pdf.rect(x,self.height-y-h,w,h,fill=1,stroke=1)
    def text(self,x,y,s,size=8,bold=False,color=INK,anchor='middle'):
        st=f'font-size: {size}px; font-family: DejaVu Sans; text-anchor: {anchor}; fill: {color}'
        if bold:st+='; font-weight: 700'
        ET.SubElement(self.group,tag('text'),{'x':str(x),'y':str(y),'style':st}).text=s
        self.color(color);self.pdf.setFont('DejaVu Sans Bold' if bold else 'DejaVu Sans',size)
        fn=self.pdf.drawCentredString if anchor=='middle' else self.pdf.drawRightString if anchor=='end' else self.pdf.drawString
        fn(x,self.height-y,s)
    def arrow(self,pts,color=LINE,lw=.7,head=3):
        p=self.pdf.beginPath();p.moveTo(pts[0][0],self.height-pts[0][1])
        for x,y in pts[1:]:p.lineTo(x,self.height-y)
        self.color(color,True);self.pdf.setLineWidth(lw);self.pdf.drawPath(p)
        ET.SubElement(self.group,tag('polyline'),{'points':' '.join(f'{x},{y}' for x,y in pts),'fill':'none','stroke':color,'stroke-width':str(lw),'stroke-linejoin':'round'})
        x,y=pts[-1];px,py=pts[-2];dx=x-px;dy=y-py;ln=(dx*dx+dy*dy)**.5;dx/=ln;dy/=ln
        tri=[(x,y),(x-head*dx+head*.5*dy,y-head*dy-head*.5*dx),(x-head*dx-head*.5*dy,y-head*dy+head*.5*dx)]
        ET.SubElement(self.group,tag('polygon'),{'points':' '.join(f'{xx},{yy}' for xx,yy in tri),'fill':color})
        p=self.pdf.beginPath();p.moveTo(tri[0][0],self.height-tri[0][1])
        for xx,yy in tri[1:]:p.lineTo(xx,self.height-yy)
        p.close();self.color(color);self.pdf.drawPath(p,fill=1,stroke=0)
    def finish(self):
        self.pdf.showPage();self.pdf.save()
        return fitz.open(stream=self.buffer.getvalue(),filetype='pdf')

def write_svg(root,path):ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)
def save_pdf(doc,i):
    doc.subset_fonts();doc.save(O/f'Figure_{i}.pdf',deflate=True,garbage=4)
    doc[0].get_pixmap(dpi=300).save(F/f'Figure_{i}.png')

for i in [1,2]:
    for ext in ['png','svg']:shutil.copy2(S/f'figures/Figure_{i}.{ext}',F/f'Figure_{i}.{ext}')
    shutil.copy2(S/f'output/pdf/Figure_{i}.pdf',O/f'Figure_{i}.pdf')

# Figure 3 legend occupies a separate row above the panel.
root=ET.parse(S/'figures/Figure_3.svg').getroot()
legend=next(e for e in root.iter() if e.get('id','').endswith('legend_1'))
texts=list(legend.iter(tag('text')))
left=min(float(re.findall(r'[-+]?\d*\.?\d+',e.get('d'))[0]) for e in legend.iter(tag('path')))
right=max(float(t.get('x'))+pdfmetrics.stringWidth(t.text,'DejaVu Sans',8) for t in texts)
dx=(W-(right-left))/2-left;dy=24
legend.set('transform',f'translate({dx} {dy})')
newh=293+dy
root.set('height',f'{newh*SCALE/72*25.4:.5f}mm');root.set('viewBox',f'0 0 {W} {newh}')
for e in root:
    if e.tag==tag('rect'):e.set('height',str(newh))
for e in root.iter(tag('clipPath')):
    if e.get('id')=='band_3_0':next(iter(e)).set('height',str(newh))
write_svg(root,F/'Figure_3.svg')
src=fitz.open(S/'output/pdf/Figure_3.pdf');page=src[0]
axis_label='Share of squared response (%)'
labels={t.text for t in texts}|{axis_label};removed=[]
for b in page.get_text('dict')['blocks']:
    for l in b.get('lines',[]):
        for s in l['spans']:
            if s['text'] in labels:
                page.add_redact_annot(fitz.Rect(s['bbox']),fill=False);removed.append(s['text'])
assert set(removed)==labels
page.apply_redactions(images=0,graphics=0,text=0)
new=fitz.open();np=new.new_page(width=W*SCALE,height=newh*SCALE)
np.show_pdf_page(fitz.Rect(0,0,page.rect.width,page.rect.height),src,0)
draw=Drawing(newh)
np.draw_rect(fitz.Rect(0,266*SCALE,W*SCALE,293*SCALE),color=None,fill=(1,1,1),overlay=True)
draw.text(361.502362,301.982719-26,axis_label,8.2)
for patch in legend.iter(tag('path')):
    vals=[float(v) for v in re.findall(r'[-+]?\d*\.?\d+',patch.get('d'))]
    xs=vals[::2];ys=vals[1::2];x0,x1=min(xs),max(xs);y0,y1=min(ys),max(ys)
    fill=re.search(r'fill: (#[0-9a-fA-F]{6})',patch.get('style')).group(1)
    draw.rect(x0+dx,y0-26+dy,x1-x0,y1-y0,fill=fill,stroke=fill,lw=.6)
for t in texts:draw.text(float(t.get('x'))+dx,float(t.get('y'))-26+dy,t.text,anchor='start')
overlay=draw.finish();np.show_pdf_page(np.rect,overlay,0);save_pdf(new,3)
new.close();overlay.close();src.close()

# FIGURE 4A: one predictor, then a common decomposition applied separately to
# prediction and measurement. The measured response is not a model input.
A_H=128
d=Drawing(A_H)
d.text(20.409449,19.867188,'A',12,True,anchor='start')
d.text(38.777953,19.140203,'Control-expression baseline and decomposition',9.2,True,anchor='start')
d.rect(20.409449,33,120,41,r=1.2)
d.text(80.409449,49,'Control expression')
d.text(80.409449,62,'X1 · 978 genes')
d.rect(170.409449,33,170,41,r=1.2)
d.text(255.409449,45,'3H2048',8.6,True)
for x in [181.409449,235.409449,289.409449]:
    d.rect(x,52,40,17,fill='#e6edf3',stroke=BORDER,r=.7)
    d.text(x+20,63.5,'2,048',7.8)
for x in [221.409449,275.409449]:d.arrow([(x+1.5,60.5),(x+12,60.5)],head=2.5)
d.rect(370.409449,33,120,41,fill='#eeebf5',r=1.2)
d.text(430.409449,49,'Predicted response')
d.text(430.409449,62,'ΔX · 978 genes')
d.arrow([(142.409449,53.5),(168.409449,53.5)])
d.arrow([(342.409449,53.5),(368.409449,53.5)])
d.rect(20.409449,98,120,25,r=1.2)
d.text(80.409449,113.5,'Measured response ΔX')
d.rect(170.409449,98,154,25,r=1.2)
d.text(247.409449,113.5,'Decompose separately')
d.arrow([(142.409449,110.5),(168.409449,110.5)])
d.arrow([(430.409449,75.5),(430.409449,86),(247.409449,86),(247.409449,96.5)])
d.arrow([(326.409449,110.5),(348.409449,110.5)])
for j,(s,c) in enumerate(zip(['μ','α','β','γ'],COLORS)):
    x=350.409449+j*37
    d.rect(x,98,29,25,fill=c,stroke=c,r=1)
    d.text(x+14.5,114,s,10,True,color='#ffffff')
    if j<3:d.text(x+33,113.5,'+',8,color=LINE)
overlay=d.finish()
root=ET.parse(S/'figures/Figure_4.svg').getroot()
height=610+A_H-100;root.set('height',f'{height*SCALE/72*25.4:.5f}mm');root.set('viewBox',f'0 0 {W} {height}')
groups=[e for e in root if e.tag==tag('g')]
root.remove(groups[0])
for e in groups[1:]:
    y=float(re.search(r'translate\(0 ([\d.]+)\)',e.get('transform')).group(1))
    e.set('transform',f'translate(0 {y+A_H-100})')
for e in root:
    if e.tag==tag('rect'):e.set('height',str(height))
root.insert(2,d.group)
write_svg(root,F/'Figure_4.svg')
src=fitz.open(S/'output/pdf/Figure_4.pdf');page=src[0]
new=fitz.open();np=new.new_page(width=W*SCALE,height=height*SCALE)
np.show_pdf_page(fitz.Rect(0,0,W*SCALE,A_H*SCALE),overlay,0)
np.show_pdf_page(fitz.Rect(0,A_H*SCALE,W*SCALE,height*SCALE),src,0,clip=fitz.Rect(0,100*SCALE,W*SCALE,page.rect.height))
save_pdf(new,4)

# Verify that schematic layout preserves numerical panels B-E.
old_clip=fitz.Rect(0,100*SCALE,W*SCALE,page.rect.height)
new_clip=fitz.Rect(0,A_H*SCALE,W*SCALE,np.rect.height)
oldcheck=fitz.open();op=oldcheck.new_page(width=old_clip.width,height=old_clip.height)
op.show_pdf_page(op.rect,src,0,clip=old_clip)
newcheck=fitz.open();xp=newcheck.new_page(width=new_clip.width,height=new_clip.height)
xp.show_pdf_page(xp.rect,new,0,clip=new_clip)
oldpix=op.get_pixmap(dpi=144);newpix=xp.get_pixmap(dpi=144)
assert (oldpix.width,oldpix.height)==(newpix.width,newpix.height)
im1=Image.frombytes('RGB',[oldpix.width,oldpix.height],oldpix.samples)
im2=Image.frombytes('RGB',[newpix.width,newpix.height],newpix.samples)
diff=ImageChops.difference(im1,im2)
maxdiff=max(v[1] for v in diff.getextrema())
assert maxdiff<=2,maxdiff
assert op.get_text()==xp.get_text()
oldcheck.close();newcheck.close();new.close();overlay.close();src.close()

