"""Lay out vector figures with explanatory notes in separate captions.

SVG text is rendered in embedded DejaVu Sans; graphical PDF objects remain
vector. Panel coordinates and axis scales are preserved."""
from pathlib import Path
import sys, io, json, math, re, copy, hashlib, collections, csv, html
import xml.etree.ElementTree as ET

import os
ROOT=Path(os.environ['TAHOE_FIGURE_WORKDIR'])
R=ROOT/'layout'
V = R.parent
P=ROOT/'base'
import pymupdf as fitz
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

F = R/'figures'; O = R/'output/pdf'; Q = R/'qa'
for directory in [F, O, Q]: directory.mkdir(parents=True, exist_ok=True)
import matplotlib
_fontdir=Path(matplotlib.get_data_path())/'fonts/ttf'
pdfmetrics.registerFont(TTFont('DejaVu Sans',str(_fontdir/'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DejaVu Sans Bold',str(_fontdir/'DejaVuSans-Bold.ttf')))
SVG = 'http://www.w3.org/2000/svg'; XLINK = 'http://www.w3.org/1999/xlink'
ET.register_namespace('', SVG); ET.register_namespace('xlink', XLINK)
tag = lambda s: '{'+SVG+'}'+s
sha = lambda b: hashlib.sha256(b).hexdigest()
TARGET_MM = 174.0

REMOVE_STARTS = {
  1: ['95% intervals', 'Tahoe: all eligible cross-plate repeats.'],
  2: ['869 genes', 'n = matched drugs', '95% intervals', 'Tahoe to external sources', 'Chance ='],
  3: ['Observed response composition', 'Native counts:', 'Residuals include measurement error;'],
  4: ['X1-only coverage and information in measured drug effects', 'Development panel',
      'Fully connected + ReLU + dropout', 'No drug input;', '\u03b2 + \u03b3: 95.9%',
      'Measured-input diagnostic;', 'Source-plate \u03b2', 'Shading: cross-plate predictions.',
      '\u03b2 uses all source-plate cells.'],
}
REPLACE = {'PCC \u00b7 95% interval': 'PCC', 'n < 5; descriptive': 'n < 5'}
# (source top, source bottom, new white gap before band), in SVG/PDF points.
BANDS = {
  1: [(0,232,0), (267.6400002413333,499.6400002413333,10)],
  2: [(0,24,0),(44,327,0),(340.3600003086667,364.3600003086667,10),
      (380.3600003086667,571.3600003086667,0)],
  3: [(26,319,0)],
  4: [(44,144,0),(159,428,0),(457.0000004166667,480.0000004166667,12),
      (503.0000004166667,709.0000004166667,0)],
}

def multiply(A, B):
    a,b,c,d,e,f=A;g,h,i,j,k,l=B
    return (a*g+c*h,b*g+d*h,a*i+c*j,b*i+d*j,a*k+c*l+e,b*k+d*l+f)

def transform(s):
    M=(1.,0.,0.,1.,0.,0.)
    for name,args in re.findall(r'(\w+)\s*\(([^)]*)\)',s or ''):
        v=[float(x) for x in re.findall(r'[-+]?(?:\d*\.)?\d+(?:[eE][-+]?\d+)?',args)]
        if name=='translate': T=(1,0,0,1,v[0],v[1] if len(v)>1 else 0)
        elif name=='scale': T=(v[0],0,0,v[1] if len(v)>1 else v[0],0,0)
        elif name=='matrix': T=tuple(v)
        elif name=='rotate':
            q=math.radians(v[0]);T=(math.cos(q),math.sin(q),-math.sin(q),math.cos(q),0,0)
            if len(v)==3:T=multiply(multiply((1,0,0,1,v[1],v[2]),T),(1,0,0,1,-v[1],-v[2]))
        else: raise ValueError((name,args))
        M=multiply(M,T)
    return M

def get_style(e):
    return {k.strip():v.strip() for k,v in (x.split(':',1) for x in e.get('style','').split(';') if ':' in x)}

def text_records(root):
    records=[]
    def visit(e,M):
        M=multiply(M,transform(e.get('transform','')))
        if e.tag==tag('text'):
            assert len(e)==0, 'Unexpected text spans'
            st=get_style(e);s=e.text or '';size=float(st['font-size'].removesuffix('px'))
            x=float(e.get('x','0'));y=float(e.get('y','0'))
            records.append({'text':s,'style':st,'size':size,'x':x,'y':y,'M':M,
                            'world_x':M[0]*x+M[2]*y+M[4], 'world_y':M[1]*x+M[3]*y+M[5]})
        for child in e:visit(child,M)
    visit(root,(1.,0.,0.,1.,0.,0.))
    return records

def geometry_digest(root):
    # Leaf graphic attributes retain source coordinates; text and wrappers excluded.
    kinds={'path','rect','circle','ellipse','line','polyline','polygon','image','use'}
    seq=[(e.tag, sorted(e.attrib.items()),e.text) for e in root.iter() if e.tag.rsplit('}',1)[-1] in kinds]
    return sha(json.dumps(seq,ensure_ascii=False).encode())

def redraw_text(root, page_width, page_height):
    buffer=io.BytesIO();c=canvas.Canvas(buffer,pagesize=(page_width,page_height),pageCompression=1)
    for r in text_records(root):
        st=r['style'];font='DejaVu Sans Bold' if st.get('font-weight') in ['bold','700'] else 'DejaVu Sans'
        size=r['size'];s=r['text'];width=pdfmetrics.stringWidth(s,font,size)
        anchor=st.get('text-anchor','start');dx=-width/2 if anchor=='middle' else -width if anchor=='end' else 0
        color=st.get('fill','#192d3a');assert re.fullmatch(r'#[0-9a-fA-F]{6}',color),color
        rgb=[int(color[i:i+2],16)/255 for i in [1,3,5]]
        c.saveState();c.transform(1,0,0,-1,0,page_height);c.transform(*r['M'])
        c.translate(r['x'],r['y']);c.scale(1,-1);c.setFillColorRGB(*rgb)
        c.setFont(font,size);c.drawString(dx,0,s);c.restoreState()
    c.showPage();c.save()
    return fitz.open(stream=buffer.getvalue(),filetype='pdf')

legends=json.loads((P/'source/LEGENDS_EN.json').read_text(encoding='utf-8'))
audit=[];moved=[];book=fitz.open();review=[];caps={}
titles={1:'Within-source repeatability',2:'Cross-source concordance',3:'Response composition',4:'Control-expression prediction'}

for i in [1,2,3,4]:
    source=P/f'plates/Figure_{i}.svg'
    root=ET.parse(source).getroot();W,H=[float(x) for x in root.get('viewBox').split()][2:]
    old_graphics=geometry_digest(root);old_text=collections.Counter(r['text'] for r in text_records(root))
    removed=[];replaced=[]
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag!=tag('text'):continue
            s=child.text or ''
            if any(s.startswith(prefix) for prefix in REMOVE_STARTS[i]):
                removed.append(s);parent.remove(child);continue
            if s in REPLACE:
                child.text=REPLACE[s];replaced.append((s,child.text))
            st=get_style(child);st['font-family']='DejaVu Sans'
            child.set('style','; '.join(k+': '+v for k,v in st.items()))
    assert geometry_digest(root)==old_graphics
    current=collections.Counter(r['text'] for r in text_records(root))
    predicted=old_text.copy()
    for s in removed:predicted[s]-=1
    for a,b in replaced:predicted[a]-=1;predicted[b]+=1
    assert +predicted==current
    bands=BANDS[i];scale=(TARGET_MM/25.4*72)/W
    new_H=sum(y1-y0+gap for y0,y1,gap in bands)
    records=text_records(root)
    for r in records:
        assert any(y0<=r['world_y']<=y1 for y0,y1,gap in bands),(i,'cropped text',r['text'],r['world_y'])
    outroot=ET.Element(tag('svg'),{'width':f'{TARGET_MM}mm','height':f'{new_H*scale/72*25.4:.5f}mm',
                                  'viewBox':f'0 0 {W} {new_H}'})
    defs=ET.SubElement(outroot,tag('defs'));srcgroup=ET.SubElement(defs,tag('g'),{'id':f'clean_source_{i}'})
    for child in list(root):srcgroup.append(copy.deepcopy(child))
    ET.SubElement(outroot,tag('rect'),{'width':str(W),'height':str(new_H),'fill':'white'})
    dest=0
    for j,(y0,y1,gap) in enumerate(bands):
        dest+=gap;hh=y1-y0;cid=f'band_{i}_{j}'
        cl=ET.SubElement(defs,tag('clipPath'),{'id':cid});ET.SubElement(cl,tag('rect'),{'x':'0','y':'0','width':str(W),'height':str(hh)})
        group=ET.SubElement(outroot,tag('g'),{'transform':f'translate(0 {dest})'})
        clipped=ET.SubElement(group,tag('g'),{'clip-path':f'url(#{cid})'})
        ET.SubElement(clipped,tag('use'),{'{'+XLINK+'}href':f'#clean_source_{i}','transform':f'translate(0 {-y0})'})
        dest+=hh
    ET.ElementTree(outroot).write(F/f'Figure_{i}.svg',encoding='utf-8',xml_declaration=True)

    srcdoc=fitz.open(P/f'plates/Figure_{i}.pdf');page=srcdoc[0]
    assert abs(page.rect.width-W)<.01 and abs(page.rect.height-H)<.01
    spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for line in b['lines'] for s in line['spans']]
    for span in spans:page.add_redact_annot(fitz.Rect(span['bbox']),fill=False)
    page.apply_redactions(images=0,graphics=0,text=0)
    assert not page.get_text().strip(), 'Source text remains'
    overlay=redraw_text(root,W,H);page.show_pdf_page(page.rect,overlay,0)
    result=fitz.open();newpage=result.new_page(width=W*scale,height=new_H*scale)
    dest=0
    for y0,y1,gap in bands:
        dest+=gap
        newpage.show_pdf_page(fitz.Rect(0,dest*scale,W*scale,(dest+y1-y0)*scale),srcdoc,0,clip=fitz.Rect(0,y0,W,y1))
        dest+=y1-y0
    result.set_metadata({'title':f'Figure {i}. {titles[i]}','creator':'Vector figure composition'})
    result.subset_fonts()
    result.save(O/f'Figure_{i}.pdf',deflate=True,garbage=4)
    newpage.get_pixmap(dpi=300).save(F/f'Figure_{i}.png')
    book.insert_pdf(result)
    final_spans=[s for b in newpage.get_text('dict')['blocks'] if 'lines' in b for line in b['lines'] for s in line['spans']]
    visible_fonts=sorted({s['font'] for s in final_spans})
    assert all('DejaVuSans' in f for f in visible_fonts),visible_fonts
    assert all('\ufffd' not in s['text'] for s in final_spans)
    minimum=min(s['size'] for s in final_spans)
    assert minimum>=7.0,(i,minimum)
    # Text must lie inside the page, allowing subpixel font metric tolerance.
    bounds=newpage.rect+(-.4,-.4,.4,.4)
    outside=[s['text'] for s in final_spans if not bounds.contains(fitz.Rect(s['bbox']))]
    assert not outside,(i,outside)
    finaltext=newpage.get_text()
    for s in removed:assert s not in finaltext,(i,'removed note retained',s)
    for s in removed:moved.append({'figure':i,'removed_from_image':s,'destination':'Figure legend'})
    audit.append({'figure':i,'removed_text':removed,'shortened_labels':replaced,
                  'all_graphical_SVG_attributes_unchanged':True,'remaining_text_values_preserved':True,
                  'font_family':visible_fonts,'minimum_font_pt':minimum,'width_mm':TARGET_MM,
                  'height_mm':new_H*scale/72*25.4,'source_svg_sha256':sha(source.read_bytes()),
                  'output_pdf_sha256':sha((O/f'Figure_{i}.pdf').read_bytes()),'output_svg_sha256':sha((F/f'Figure_{i}.svg').read_bytes()),
                  'source_crop_bands':bands,'retained_text_nodes':len(records),'text_outside_canvas':outside})
    result.close();overlay.close();srcdoc.close()

book.close()
