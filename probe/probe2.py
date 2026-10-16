import requests, re
from bs4 import BeautifulSoup
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36 Edg/153.0.0.0"
H = {"User-Agent":UA,"Accept-Language":"fa-IR,fa;q=0.9,en;q=0.8"}
sites = {
 "zoomit.ir":"https://www.zoomit.ir/",
 "isna.ir":"https://www.isna.ir/",
 "tabnak.ir":"https://www.tabnak.ir/",
 "khabaronline.ir":"https://www.khabaronline.ir/",
 "hamshahri":"https://www.hamshahrionline.ir/",
 "digiato.com":"https://digiato.com/",
 "varzesh3.com":"https://www.varzesh3.com/",
}
for name,url in sites.items():
    try:
        r = requests.get(url, headers=H, timeout=15)
        r.encoding = r.apparent_encoding or r.encoding
        soup = BeautifulSoup(r.text,"html.parser")
        pats = re.compile(r"(news|article|خبر|اخبار|/fa/news|/روزنامه|post|blog|/tech)", re.I)
        links=[]
        for a in soup.find_all("a", href=True):
            h=a["href"]
            if pats.search(h) and len(h)>15:
                if h.startswith("/"): h="https://"+url.split("//")[1].split("/")[0]+h
                links.append(h)
        uniq=[]
        for l in links:
            if l not in uniq: uniq.append(l)
        print(f"=== {name} ({len(uniq)} candidate links) ===")
        for l in uniq[:5]: print("   ", l[:120])
        # comment markers in HTML
        low=r.text.lower()
        markers=["disqus","comment","دیدگاه","نظر","کامنت","doreh","giscus","hypercomments","cackle"]
        found=[m for m in markers if m in low]
        print("   markers:", found)
    except Exception as e:
        print(f"=== {name} ERROR {type(e).__name__}: {str(e)[:100]}")
