import requests, re, json
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36 Edg/153.0.0.0"
H = {"User-Agent": UA, "Accept-Language": "fa-IR,fa;q=0.9,en;q=0.8"}

def show(name, url, ctx=None):
    try:
        r = requests.get(url, headers=H, timeout=15, allow_redirects=True)
        r.encoding = r.apparent_encoding or r.encoding
        print(f"\n===== {name} {r.status_code} len={len(r.content)} ct={r.headers.get('content-type')}")
        print("  final:", r.url[:110])
        return r
    except Exception as e:
        print(f"\n===== {name} ERROR {type(e).__name__}: {str(e)[:120]}")
        return None

# ---- ZOOMIT: look at HTML around comments ----
r = show("zoomit article", "https://www.zoomit.ir/ai-articles/467836-jensen-huang-ai-energy-climate-change-controversy/")
if r:
    t = r.text
    for kw in ["comments", "comment", "دیدگاه", "کامنت"]:
        for m in re.finditer(kw, t, re.I):
            s = max(0, m.start()-150); e = min(len(t), m.end()+250)
            seg = t[s:e].replace("\n", " ")
            print(f"  [{kw}] ...{seg[:380]}...")
            break

# ---- TABNAK ----
r = show("tabnak article", "https://www.tabnak.ir/fa/news/1395983/")
if r:
    t = r.text
    for kw in ["comment", "/comment", "نظر"]:
        idxs = [m.start() for m in re.finditer(kw, t, re.I)][:2]
        for i in idxs:
            print(f"  [{kw}] ...{t[max(0,i-120):i+220]}...".replace("\n"," ")[:400])

# ---- KHABARONLINE ----
r = show("khabaronline article", "https://www.khabaronline.ir/news/1940521/")
if r:
    print("  len text:", len(r.text))
    for kw in ["comment", "نظر", "دیدگاه", "کامنت", "api"]:
        idxs = [m.start() for m in re.finditer(kw, r.text, re.I)][:2]
        print(f"  [{kw}] count={len(re.findall(kw, r.text, re.I))}")
        for i in idxs:
            print(f"     ...{r.text[max(0,i-100):i+180]}...".replace("\n"," ")[:330])

# ---- HAMSH AHRI ----
r = show("hamshahri article", "https://www.hamshahrionline.ir/news/531930/")
if r:
    for kw in ["comment", "نظر", "دیدگاه", "کامنت"]:
        idxs = [m.start() for m in re.finditer(kw, r.text, re.I)][:2]
        print(f"  [{kw}] count={len(re.findall(kw, r.text, re.I))}")
        for i in idxs:
            print(f"     ...{r.text[max(0,i-100):i+220]}...".replace("\n"," ")[:360])
