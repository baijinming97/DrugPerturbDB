import sys,json
from rdkit import Chem,RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*')
allowed=['Cl','[Cl-]','Br','[Br-]','[Na+]','[K+]','[Ca+2]','[Mg+2]','O','O=S(=O)(O)O','O=S(=O)([O-])O','O=S(=O)([O-])[O-]','CS(=O)(=O)O','CS(=O)(=O)[O-]','CC(=O)O','CC(=O)[O-]']
allowed+=['O=C(O)CC(O)C(=O)O','O=C(O)C[C@H](O)C(=O)O','O=C(O)C[C@@H](O)C(=O)O','O=C([O-])CC(O)C(=O)O','O=C([O-])C[C@H](O)C(=O)O','O=C([O-])C[C@@H](O)C(=O)O','O=C(O)/C=C/C(=O)O','O=C(O)/C=C\\C(=O)O','O=C(O)C=CC(=O)O','O=C([O-])C=CC(=O)O','O=C([O-])/C=C/C(=O)O','O=C([O-])/C=C\\C(=O)O','Cc1ccc(S(=O)(=O)O)cc1','Cc1ccc(S(=O)(=O)[O-])cc1']
allowed={Chem.MolToSmiles(Chem.MolFromSmiles(s),isomericSmiles=True) for s in allowed}
out=[];unch=rdMolStandardize.Uncharger()
for z in json.load(sys.stdin):
 z['parent_key']='';z['parent_status']='unparseable';m=Chem.MolFromSmiles(z['smiles'])
 if m:
  z['computed_key']=Chem.MolToInchiKey(m)
  if z['computed_key']!=z['key']:z['parent_status']='source SMILES does not reproduce supplied stereospecific key'
  else:
   fs=sorted(Chem.GetMolFrags(m,asMols=True,sanitizeFrags=True),key=lambda x:x.GetNumHeavyAtoms(),reverse=True)
   if all(Chem.MolToSmiles(f,isomericSmiles=True) in allowed for f in fs[1:]):
    z['parent_key']=Chem.MolToInchiKey(unch.uncharge(fs[0]));z['parent_status']='verified stereochemistry-preserving neutral parent'
   else:z['parent_status']='unapproved additional component; mixture not split'
 z.pop('smiles',None);out.append(z)
json.dump(out,sys.stdout)
