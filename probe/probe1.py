import requests, sys
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36 Edg/153.0.0.0"
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
        r = requests.get(url, headers={"User-Agent":UA,"Accept-Language":"fa-IR,fa;q=0.9,en;q=0.8"}, timeout=15, allow_redirects=True)
        print(f"{name:18} {r.status_code} final={r.url[:70]} len={len(r.content)} enc={r.encoding!r} server={r.headers.get('server','?')}")
    except Exception as e:
        print(f"{name:18} ERROR {type(e).__name__}: {str(e)[:110]}")
