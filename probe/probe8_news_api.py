"""Probe Persian news sites for a COMMENTS JSON endpoint (best) or SSR comments."""
import requests, re, json

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36 Edg/153.0.0.0")
H = {"User-Agent": UA, "Accept-Language": "fa-IR,fa;q=0.9,en;q=0.8",
     "Accept": "application/json, text/plain, */*"}


def probe_json(name, url):
    try:
        r = requests.get(url, headers=H, timeout=20)
        ct = r.headers.get("content-type", "")
        print(f"  {name}: HTTP {r.status_code} ct={ct[:40]} len={len(r.content)}")
        print(f"     body[:400]={r.text[:400]!r}")
        if "json" in ct:
            try:
                j = r.json()
                print(f"     JSON {type(j).__name__}")
                if isinstance(j, dict):
                    print(f"     keys={list(j.keys())[:12]}")
                if isinstance(j, list):
                    print(f"     list len={len(j)} first={str(j[0])[:300] if j else None}")
            except Exception as e:
                print(f"     json parse fail {e}")
        return r
    except Exception as e:
        print(f"  {name}: FAIL {type(e).__name__}: {str(e)[:130]}")
        return None


print("################ VARZESH3 ################")
for nid in ("2431742",):
    probe_json("v3 comments /v1.0", f"https://web-api.varzesh3.com/v1.0/news/{nid}/comments")
    probe_json("v3 comments ?page=1", f"https://web-api.varzesh3.com/v1.0/news/{nid}/comments?page=1")
    probe_json("v3 comments /v2.0", f"https://web-api.varzesh3.com/v2.0/news/{nid}/comments")
    probe_json("v3 news detail api", f"https://web-api.varzesh3.com/v1.0/news/{nid}")
    probe_json("v3 home/latest", "https://web-api.varzesh3.com/v1.0/home/latest")

print("\n################ TABNAK ################")
for u in ("https://www.tabnak.ir/fa/news/1395983/",
          "https://www.tabnak.ir/fa/news/1395983/%D8%AE%D8%A8%D8%B1"):
    try:
        r = requests.get(u, headers={"User-Agent": UA}, timeout=20)
        r.encoding = "utf-8"
        t = r.text
        print(f"  page {u} -> {r.status_code} len={len(t)}")
        for kw in ["box_news_comment", "x-template", "comment", "comments.php", "نظر"]:
            print(f"     kw {kw!r}: {len(re.findall(kw, t, re.I))}")
        for m in re.finditer(r"[\"'(]([^\"'()\s]{0,90}(?:comment|comments)[^\"'()\s]{0,90})[\"')]", t, re.I):
            print("     EP:", m.group(1)[:140])
        # print the x-template comment box
        m = re.search(r"box_news_comment[^>]*>", t)
        if m:
            print("     --- comment box region ---")
            print("     ", t[m.start():m.start() + 1200].replace("\n", " ")[:1200])
    except Exception as e:
        print(f"  page FAIL {type(e).__name__}: {str(e)[:130]}")

print("\n################ ISNA ################")
for u in ("https://www.isna.ir/", "https://www.isna.ir/news/1405060600001/"):
    try:
        r = requests.get(u, headers={"User-Agent": UA}, timeout=20)
        r.encoding = "utf-8"
        print(f"  {u} -> {r.status_code} len={len(r.text)}")
        for kw in ["comment", "نظر", "دیدگاه"]:
            print(f"     kw {kw!r}: {len(re.findall(kw, r.text, re.I))}")
    except Exception as e:
        print(f"  {u} FAIL {type(e).__name__}: {str(e)[:130]}")
