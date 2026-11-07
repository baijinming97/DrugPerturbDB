import zipfile,re,xml.etree.ElementTree as ET
NS='{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
def colnum(ref):
 n=0
 for c in re.sub('[0-9]','',ref):n=26*n+ord(c)-64
 return n-1
def rows(path):
 with zipfile.ZipFile(path) as z:
  ss=[''.join(e.itertext()) for e in ET.fromstring(z.read('xl/sharedStrings.xml'))] if 'xl/sharedStrings.xml' in z.namelist() else []
  with z.open('xl/worksheets/sheet1.xml') as f:
   for _,row in ET.iterparse(f,events=('end',)):
    if row.tag!=NS+'row':continue
    vals={}
    for c in row:
     v=c.find(NS+'v');t=c.get('t')
     val=''.join(x.text or '' for x in c.iter(NS+'t')) if t=='inlineStr' else '' if v is None else ss[int(v.text)] if t=='s' else v.text
     vals[colnum(c.get('r'))]=val
    row.clear();yield vals
def frame(path):
 import pandas as pd
 rs=rows(path);head=None;out=[]
 for r in rs:
  if head is None:
   if len(r)>3:head=r
  else:out.append({head[k]:v for k,v in r.items() if k in head})
 assert head is not None,path
 return pd.DataFrame(out)
