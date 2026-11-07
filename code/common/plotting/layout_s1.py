from pathlib import Path
import os, io, re, math, hashlib, copy, json
import xml.etree.ElementTree as ET
import pymupdf as fitz
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
ROOT=Path(os.environ['TAHOE_FIGURE_WORKDIR']);R=ROOT/'supplement';R.mkdir(exist_ok=True)
(R/'figures').mkdir(exist_ok=True)
SVG='http://www.w3.org/2000/svg';XLINK='http://www.w3.org/1999/xlink'
ET.register_namespace('',SVG);ET.register_namespace('xlink',XLINK)
tag=lambda s:'{'+SVG+'}'+s
sha=lambda b:hashlib.sha256(b).hexdigest()
import matplotlib
_fontdir=Path(matplotlib.get_data_path())/'fonts/ttf'
pdfmetrics.registerFont(TTFont('DejaVu Sans',str(_fontdir/'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DejaVu Sans Bold',str(_fontdir/'DejaVuSans-Bold.ttf')))
def multiply(A, B):
    a, b, c, d, e, f = A
    g, h, i, j, k, l = B
    return (a * g + c * h, b * g + d * h, a * i + c * j, b * i + d * j, a * k + c * l + e, b * k + d * l + f)

def transform(s):
    M = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    for name, args in re.findall('(\\w+)\\s*\\(([^)]*)\\)', s or ''):
        v = [float(x) for x in re.findall('[-+]?(?:\\d*\\.)?\\d+(?:[eE][-+]?\\d+)?', args)]
        if name == 'translate':
            T = (1, 0, 0, 1, v[0], v[1] if len(v) > 1 else 0)
        elif name == 'scale':
            T = (v[0], 0, 0, v[1] if len(v) > 1 else v[0], 0, 0)
        elif name == 'matrix':
            T = tuple(v)
        elif name == 'rotate':
            q = math.radians(v[0])
            T = (math.cos(q), math.sin(q), -math.sin(q), math.cos(q), 0, 0)
            if len(v) == 3:
                T = multiply(multiply((1, 0, 0, 1, v[1], v[2]), T), (1, 0, 0, 1, -v[1], -v[2]))
        else:
            raise ValueError((name, args))
        M = multiply(M, T)
    return M

def get_style(e):
    return {k.strip(): v.strip() for k, v in (x.split(':', 1) for x in e.get('style', '').split(';') if ':' in x)}

def text_records(root):
    records = []

    def visit(e, M):
        M = multiply(M, transform(e.get('transform', '')))
        if e.tag == tag('text'):
            assert len(e) == 0, 'Unexpected text spans'
            st = get_style(e)
            s = e.text or ''
            size = float(st['font-size'].removesuffix('px'))
            x = float(e.get('x', '0'))
            y = float(e.get('y', '0'))
            records.append({'text': s, 'style': st, 'size': size, 'x': x, 'y': y, 'M': M, 'world_x': M[0] * x + M[2] * y + M[4], 'world_y': M[1] * x + M[3] * y + M[5]})
        for child in e:
            visit(child, M)
    visit(root, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0))
    return records

def geometry_digest(root):
    kinds = {'path', 'rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon', 'image', 'use'}
    seq = [(e.tag, sorted(e.attrib.items()), e.text) for e in root.iter() if e.tag.rsplit('}', 1)[-1] in kinds]
    return sha(json.dumps(seq, ensure_ascii=False).encode())

def redraw_text(root, page_width, page_height):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=(page_width, page_height), pageCompression=1)
    for r in text_records(root):
        st = r['style']
        font = 'DejaVu Sans Bold' if st.get('font-weight') in ['bold', '700'] else 'DejaVu Sans'
        size = r['size']
        s = r['text']
        width = pdfmetrics.stringWidth(s, font, size)
        anchor = st.get('text-anchor', 'start')
        dx = -width / 2 if anchor == 'middle' else -width if anchor == 'end' else 0
        color = st.get('fill', '#192d3a')
        assert re.fullmatch('#[0-9a-fA-F]{6}', color), color
        rgb = [int(color[i:i + 2], 16) / 255 for i in [1, 3, 5]]
        c.saveState()
        c.transform(1, 0, 0, -1, 0, page_height)
        c.transform(*r['M'])
        c.translate(r['x'], r['y'])
        c.scale(1, -1)
        c.setFillColorRGB(*rgb)
        c.setFont(font, size)
        c.drawString(dx, 0, s)
        c.restoreState()
    c.showPage()
    c.save()
    return fitz.open(stream=buffer.getvalue(), filetype='pdf')
srcsvg=ROOT/'base/plates/Figure_S1.svg';srcpdf=ROOT/'base/plates/Figure_S1.pdf'
root=ET.parse(srcsvg).getroot();W,H=map(float,root.get('viewBox').split()[2:]);before=geometry_digest(root)
remove=['Identical repeat pairs','Gray lines join','Matched cell line','Retained 95%','MIX-seq has one repeat drug']
removed=[]
for parent in list(root.iter()):
    for child in list(parent):
        if child.tag!=tag('text'):continue
        if any((child.text or '').startswith(s) for s in remove):removed.append(child.text);parent.remove(child);continue
        st=get_style(child);st['font-family']='DejaVu Sans';child.set('style','; '.join(k+': '+v for k,v in st.items()))
assert len(removed)==5 and geometry_digest(root)==before
records=text_records(root);cropH=409.0;scale=(174/25.4*72)/W
assert max(r['world_y'] for r in records)<cropH
root.set('viewBox',f'0 0 {W} {cropH}');root.set('width','174mm');root.set('height',f'{cropH*scale/72*25.4}mm')
ET.ElementTree(root).write(R/'figures/Figure_S1.svg',encoding='utf-8',xml_declaration=True)
src=fitz.open(srcpdf);page=src[0]
for b in page.get_text('dict')['blocks']:
    for ln in b.get('lines',[]):
        for s in ln['spans']:page.add_redact_annot(fitz.Rect(s['bbox']),fill=False)
page.apply_redactions(images=0,graphics=0,text=0);assert not page.get_text().strip()
overlay=redraw_text(root,W,H);page.show_pdf_page(page.rect,overlay,0)
clean=fitz.open();p=clean.new_page(width=W*scale,height=cropH*scale)
p.show_pdf_page(p.rect,src,0,clip=fitz.Rect(0,0,W,cropH));clean.subset_fonts()
clean.save(R/'figures/Figure_S1.pdf',deflate=True,garbage=4);p.get_pixmap(dpi=220).save(R/'figures/Figure_S1.png')
assert all(s not in p.get_text() for s in removed)
