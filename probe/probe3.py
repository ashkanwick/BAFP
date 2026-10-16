import requests, re
from bs4 import BeautifulSoup
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36 Edg/153.0.0.0"
H = {"User-Agent":UA,"Accept-Language":"fa-IR,fa;q=0.9,en;q=0.8"}
arts = {
 "zoomit": "https://www.zoomit.ir/ai-articles/467836-jensen-huang-ai-energy-climate-change-controversy/",
 "tabnak": "https://www.tabnak.ir/fa/news/1395983/",
 "khabaronline": "https://www.khabaronline.ir/news/1940521/",
 "hamshahri": "https://www.hamshahrionline.ir/news/531930",
 "digiato": "https://digiato.com/topic/tech",
 "varzesh3": "https://www.varzesh3.com/news/2431742/",
}
for name,url in arts.items():
    try:
        r=requests.get(url,headers=H,timeout=15,allow_redirects=True)
        r.encoding = r.apparent_encoding or r.encoding
        print(f"\n===== {name} {r.status_code} len={len(r.content)} final={r.url[:90]}")
        low=r.text.lower()
        for m in ["disqus","comment_count","comments","دیدگاه","نظر","کامنت","api/comment","/comment","data-comment","giscus","hypercomments","talk","graphcomment"]:
            c=low.count(m)
            if c: print(f"   marker {m!r}: {c}")
        # look for api/json endpoints in page
        eps=set(re.findall(r'https?://[^"\'\s<>]{0,80}(?:comment|نظر)[^"\'\s<>]{0,80}', r.text, re.I))
        for e in list(eps)[:10]: print("   EP:", e[:130])
    except Exception as e:
        print(f"\n===== {name} ERROR {type(e).__name__}: {str(e)[:110]}")
