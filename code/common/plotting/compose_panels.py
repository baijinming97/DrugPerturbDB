"""Assemble four main and two supplemental figures from vector panels."""
from pathlib import Path
import sys,copy,json,hashlib,base64,html,xml.etree.ElementTree as ET
R=Path(__file__).resolve().parents[1];V=R.parent
import pymupdf
F=R/'figures';P=R/'plates';P.mkdir(exist_ok=True)
OUT=R/'output/pdf';OUT.mkdir(parents=True,exist_ok=True)
Q=R/'qa/figure_book';Q.mkdir(parents=True,exist_ok=True)
groups={'Figure_1':['fig1a_expression','fig1b_response'],
 'Figure_2':['fig2_ab_similarity','fig2_cd_discrimination'],
 'Figure_3':['fig3a_full'],'Figure_4':['fig4_focus','FIG4E_CROSS_PLATE_LINK'],
 'Figure_S1':['figS1_robustness']}
leg=json.loads((R/'source/LEGENDS_EN.json').read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
ns='http://www.w3.org/2000/svg';ET.register_namespace('',ns);ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
TW=180/25.4*72;gap=7;book=pymupdf.open();manifest=[];sections=[];inline={}
titles=['Within-source repeatability','Cross-source concordance and drug discrimination',
 'Observed response composition','Control-only diagnostic and measured-input information',
 'Robustness of measurement comparisons','Association and complete-response utility']
for (name,stems),title in zip(groups.items(),titles):
    dims=[]
    for stem in stems:
        el=ET.parse(F/f'{stem}.svg').getroot();vb=[float(x) for x in el.attrib['viewBox'].split()];dims.append(vb[2:4])
        inline[stem]={'source':str(F/f'{stem}.svg'),**leg[stem]}
    heights=[TW*h/w for w,h in dims];height=sum(heights)+gap*(len(stems)-1)
    svg=ET.Element('{'+ns+'}svg',{'width':'180mm','height':f'{height/72*25.4:.4f}mm','viewBox':f'0 0 {TW} {height}'})
    ET.SubElement(svg,'{'+ns+'}rect',{'width':str(TW),'height':str(height),'fill':'white'})
    pdf=pymupdf.open();page=pdf.new_page(width=TW,height=height);y=0
    for idx,(stem,(w,h),hh) in enumerate(zip(stems,dims,heights)):
        el=ET.parse(F/f'{stem}.svg').getroot();prefix=f'{name}_{idx}_'
        ids={n.attrib['id']:prefix+n.attrib['id'] for n in el.iter() if 'id' in n.attrib}
        for n in el.iter():
            for k,v in list(n.attrib.items()):
                if k=='id':n.set(k,ids[v]);continue
                for a,b in ids.items():
                    v=v.replace('url(#'+a+')','url(#'+b+')')
                    if v=='#'+a:v='#'+b
                n.set(k,v)
        g=ET.SubElement(svg,'{'+ns+'}g',{'transform':f'translate(0 {y}) scale({TW/w})'})
        for n in list(el):g.append(copy.deepcopy(n))
        with pymupdf.open(F/f'{stem}.pdf') as src:page.show_pdf_page(pymupdf.Rect(0,y,TW,y+hh),src,0)
        y+=hh+gap
    ET.ElementTree(svg).write(P/f'{name}.svg',encoding='utf-8',xml_declaration=True)
    pdf.set_metadata({'title':name.replace('_',' '),'creator':'Retained-estimate concise figure assembly'})
    pdf.save(P/f'{name}.pdf',deflate=True,garbage=4);page.get_pixmap(dpi=300).save(P/f'{name}.png')
    book.insert_pdf(pdf);pdf.close()
    manifest.append({'figure':name,'panels':stems,'width_mm':180,'height_mm':height/72*25.4,
      'pdf_vector':True,'sha256':{e:sha(P/f'{name}.{e}') for e in ['svg','pdf','png']}})
    url='data:image/svg+xml;base64,'+base64.b64encode((P/f'{name}.svg').read_bytes()).decode()
    caps=''.join('<p><b>'+html.escape(leg[s]['label'])+'</b>. '+html.escape(leg[s]['caption'])+'</p>' for s in stems)
    sections.append(f'<section id="{name}"><h2>{name.replace("_"," ")}. {title}</h2><img src="{url}" alt="{name}"><p class="files"><a href="plates/{name}.pdf">Vector PDF</a> · <a href="plates/{name}.svg">Editable SVG</a> · <a href="plates/{name}.png">300 dpi PNG</a></p><details><summary>Legend</summary>{caps}</details></section>')

book.close()
