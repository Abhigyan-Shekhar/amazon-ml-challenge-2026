"""Compact, country-agnostic lexical blocking keys. No labels or external data."""
from itertools import combinations
from ..normalize import normalize


def anchor_tokens(name, df):
    return sorted((t for t in set(name.split()) if len(t)>2),key=lambda t:(df.get(t,0),t))[:2]


def query_keys(name,address,df):
    rare=anchor_tokens(name,df)
    keys={('n',name)}
    if len(rare)==2: keys.add(('p',*sorted(rare)))
    addr=set(address.split())
    numbers=sorted(t for t in addr if t.isdigit())[:3]
    words=sorted((t for t in addr if len(t)>=5 and not any(c.isdigit() for c in t)),key=lambda t:(-len(t),t))[:2]
    for token in rare[:1]:
        for a in numbers+words: keys.add(('a',token,a))
    return keys


def target_keys(name,address):
    nt=sorted(t for t in set(name.split()) if len(t)>2)
    keys={('n',name)}
    keys.update(('p',a,b) for a,b in combinations(nt,2))
    address_tokens={t for t in address.split() if t.isdigit() or (len(t)>=5 and not any(c.isdigit() for c in t))}
    keys.update(('a',n,a) for n in nt for a in address_tokens)
    return keys


def address_query_keys(address,df):
    ts=set(address.split())
    words=sorted((t for t in ts if len(t)>=3 and not t.isdigit()),key=lambda t:(df.get(t,0),t))[:2]
    numbers=sorted(t for t in ts if t.isdigit())[:3]
    keys={('z',address)} if address else set()
    if len(words)==2: keys.add(('w',*sorted(words)))
    for w in words[:1]:
        for n in numbers: keys.add(('d',w,n))
    return keys


def address_target_keys(address):
    ts=set(address.split());words=sorted(t for t in ts if len(t)>=3 and not t.isdigit());numbers=[t for t in ts if t.isdigit()]
    keys={('z',address)} if address else set()
    keys.update(('w',a,b) for a,b in combinations(words,2))
    keys.update(('d',w,n) for w in words for n in numbers)
    return keys

# Small legal-form vocabulary for a retrieval view only; raw/normalized fields stay intact.
LEGAL_FORMS=frozenset({'limited','private','pvt','ltd','llc','inc','incorporated','corp','corporation','llp','co','company'})
def exact_keys(name,address):
    core=tuple(sorted(set(name.split())-LEGAL_FORMS))
    keys={('e',core)} if core else {('n',name)}
    if address: keys.add(('x',tuple(sorted(set(address.split())))))
    return keys
