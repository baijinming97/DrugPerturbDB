"""Apply the manuscript supplement reference to the Figure 4 caption."""
from pathlib import Path
import sys,json,hashlib
R=Path(__file__).resolve().parents[1];V=R.parent
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
original=Figure.savefig
def save(self,path,*args,**kwargs):
    p=Path(path);result=original(self,path,*args,**kwargs)
    if p.suffix=='.svg':
        kw=dict(kwargs);kw.pop('dpi',None);original(self,p.with_suffix('.pdf'),format='pdf',**kw)
    return result
Figure.savefig=save
p=R/'code/render_model_diagnostic.py'
src=p.read_text(encoding='utf-8').split('# S3: aligned evidence matrix',1)[0]
src=src.replace('V=O.parent','V=O/"evidence_inputs"')
ns={'__file__':str(p),'__name__':'__main__'}
try:exec(compile(src,str(p),'exec'),ns)
finally:Figure.savefig=original
prior=json.loads((R/'expected/FIG4_S34_AUDIT.json').read_text(encoding='utf-8'))
for stem,rows in ns['plotted'].items():assert rows==prior['plotted_values'][stem],stem
print('Model diagnostic panels rendered from verified tables.')
