import ast, subprocess, sys, pathlib
root = pathlib.Path('.')
files = subprocess.check_output(['git','ls-files','*.py'],text=True).split()
files += ['scripts/soma_all/provenance.py']
files = sorted(set(f for f in files if not f.startswith('tests/')))

def first(doc, n=2):
    if not doc: return ''
    paras = doc.strip().split('\n\n')[0]
    s = ' '.join(l.strip() for l in paras.splitlines())
    return s if len(s) < 300 else s[:297]+'...'

def sig(fn):
    try:
        a = ast.unparse(fn.args)
    except Exception: a = '...'
    r = f" -> {ast.unparse(fn.returns)}" if fn.returns else ''
    return f"{fn.name}({a}){r}"

def section(f):
    p = f.replace(chr(92),'/')
    src = pathlib.Path(p).read_text(encoding='utf-8')
    try: tree = ast.parse(src)
    except SyntaxError: return None
    out = [f"### `{p}`", ""]
    md = first(ast.get_docstring(tree), 1)
    if md: out += [md, ""]
    items = []
    for n in tree.body:
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and not n.name.startswith('_'):
            s = sig(n)
            if len(s) > 160: s = s[:157]+'...)'
            items.append(f"- `{s}` â€” {first(ast.get_docstring(n)) or '(no docstring)'}")
        elif isinstance(n, ast.ClassDef) and not n.name.startswith('_'):
            items.append(f"- **class `{n.name}`** â€” {first(ast.get_docstring(n)) or '(no docstring)'}")
            for m in n.body:
                if isinstance(m, ast.FunctionDef) and (not m.name.startswith('_') ) :
                    s = sig(m).replace('self, ','').replace('(self)','()')
                    if len(s) > 140: s = s[:137]+'...)'
                    items.append(f"  - `.{s}` â€” {first(ast.get_docstring(m)) or ''}")
    if not items and not md: return None
    return "\n".join(out + items + [""])

groups = {}
for f in files:
    p = f.replace(chr(92),'/')
    if p.startswith('vicon2mano/'): g = 'Package: ' + ('/'.join(p.split('/')[:-1]))
    elif p.startswith('scripts/'): g = 'Scripts: ' + '/'.join(p.split('/')[:2])
    else: g = 'Top level'
    s = section(f)
    if s: groups.setdefault(g, []).append(s)
body = []
for g in sorted(groups, key=lambda x:(x.startswith('Scripts'), x.startswith('Top'), x)):
    body.append(f"## {g}\n")
    body += groups[g]
out = pathlib.Path('docs/API_REFERENCE.md')
M = '<!-- GENERATED BELOW: do not edit -->'
head = out.read_text(encoding='utf-8').split(M)[0]
NL = chr(10)
out.write_text(head + M + NL + NL + NL.join(body), encoding='utf-8')
