"""Verify editorial asset fidelity and the self-contained LaTeX build."""
from pathlib import Path
import hashlib,json,re,sys
root=Path(__file__).resolve().parents[1]
s=(root/'main.tex').read_text()
tables=[root/(rel+'.tex') for rel in re.findall(r'\\input\{([^}]+)\}',s)]
combined=s+'\n'+'\n'.join(p.read_text() for p in tables)
labels=re.findall(r'\\label\{([^}]+)\}',combined)
assert len(labels)==len(set(labels)), 'duplicate labels'
assert not set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',combined))-set(labels),'missing label'
keys={x.strip() for c in re.findall(r'\\cite(?:\[[^]]*\])?\{([^}]+)\}',s) for x in c.split(',')}
assert not keys-set(re.findall(r'\\bibitem\{([^}]+)\}',s)),'missing citation'
assert len(re.findall(r'^\\section\{',s,re.M))==7,'expected six sections and one appendix'
for word in ['production','historical','withdrawn','legacy','v5','v6','v7','gate','gates']:
 assert not re.search(r'\b'+word+r'\b',combined,re.I),'internal wording: '+word
def values(t,key):
 out=[]
 for block in re.findall(r'\\midrule([\s\S]*?)\\bottomrule',t):
  for row in block.splitlines():
   if ' & ' not in row:continue
   if key=='repaired_pod' and row.startswith('Euclidean'):continue
   cells=row.removesuffix('\\\\').split(' & ')
   if key in ['corner','financial']:cells=cells[1:]
   if key=='assembly':cells=cells[3:]
   if key=='scaling':
    if len(cells)==11:cells=[cells[1],*cells[3:]]
    elif len(cells)==7:cells=cells[2:]
    elif len(cells)==6:cells=[cells[1],*cells[3:]]
   out+=re.findall(r'(?<![A-Za-z])\d+(?:\.\d+)?(?:e[-+]?\d+)?',' '.join(cells))
 return sorted(out)
for p in tables:
 key=p.stem.removeprefix('tab_')
 assert values((root/'provenance/frozen_tables'/p.name).read_text(),key)==values(p.read_text(),key),'numerical-cell mismatch: '+key
record=json.loads((root/'verification/asset_fidelity.json').read_text())
for rel,sha in record['unchanged_figures'].items():
 assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==sha,'figure mismatch: '+rel
for rel,sha in record['frozen_tables'].items():
 assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==sha,'frozen input mismatch: '+rel
for f in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',s):assert (root/f).is_file(),f
log=root/'verification/latex_final.log'
if log.exists():
 t=log.read_text(errors='replace')
 for x in ['undefined references','undefined on input','Overfull \\hbox','Overfull \\vbox','Label(s) may have changed','Rerun to get']:
  assert x not in t,'compile issue: '+x
print('PASS: retained numerical cells, frozen inputs, figures, six sections + appendix, wording, references and build log.')
print('This verifies the editorial bundle; it does not rerun the numerical campaign.')
