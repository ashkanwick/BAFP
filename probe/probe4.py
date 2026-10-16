import requests, json
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36 Edg/153.0.0.0"
H = {"User-Agent": UA, "Accept-Language": "fa-IR,fa;q=0.9", "Accept": "application/json, text/plain, */*"}

# varzesh3 comments API
for u in ["https://web-api.varzesh3.com/v1.0/news/2431742/comments",
          "https://web-api.varzesh3.com/v1.0/news/2431742/comments?page=1"]:
    try:
        r = requests.get(u, headers=H, timeout=15)
        print("VARZ", r.status_code, u, "ct=", r.headers.get("content-type"), "len=", len(r.content))
        print("  body[:900]:", r.text[:900].replace("\n", " "))
        try:
            j = r.json()
            print("  JSON type:", type(j).__name__)
            if isinstance(j, dict):
                for k, v in j.items():
                    print("   ", k, "=", str(v)[:250])
            elif isinstance(j, list):
                print("  list len", len(j), "first:", str(j[0])[:400] if j else None)
        except Exception as e:
            print("  notjson:", e)
    except Exception as e:
        print("VARZ ERR", type(e).__name__, str(e)[:120])

print()
for u in ["https://digiato.com/comments/feed"]:
    try:
        r = requests.get(u, headers=H, timeout=15)
        print("DIG", r.status_code, u, "len", len(r.content), "ct", r.headers.get("content-type"))
        print("  body[:800]:", r.text[:800].replace("\n", " "))
    except Exception as e:
        print("DIG ERR", type(e).__name__, str(e)[:120])
