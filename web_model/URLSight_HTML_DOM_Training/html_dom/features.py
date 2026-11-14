import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def host(url):
    try: return (urlparse(url).hostname or "").lower().strip(".")
    except Exception: return ""

def related(a,b):
    return bool(a and b) and (a==b or a.endswith("."+b) or b.endswith("."+a))

def extract_dom_features(html, page_url=""):
    soup=BeautifulSoup(html or "", "lxml"); ph=host(page_url)
    anchors=soup.find_all("a",href=True)
    a_urls=[urljoin(page_url,(a.get("href") or "").strip()) for a in anchors]
    a_hosts=[host(x) for x in a_urls if host(x)]
    ext_a=[h for h in a_hosts if ph and not related(h,ph)]
    forms=soup.find_all("form")
    actions=[urljoin(page_url,(f.get("action") or "").strip()) for f in forms]
    action_hosts=[host(x) for x in actions if host(x)]
    inputs=soup.find_all("input")
    passwords=soup.find_all("input",attrs={"type":re.compile("^password$",re.I)})
    hidden=soup.find_all("input",attrs={"type":re.compile("^hidden$",re.I)})
    scripts=soup.find_all("script"); images=soup.find_all("img")
    script_hosts=[host(urljoin(page_url,s["src"])) for s in scripts if s.get("src")]
    image_hosts=[host(urljoin(page_url,i["src"])) for i in images if i.get("src")]
    external=lambda hs: sum(1 for h in hs if ph and not related(h,ph))
    f={
      "html_length":len(html or ""), "visible_text_length":len(soup.get_text(" ",strip=True)),
      "title_length":len(soup.title.get_text(" ",strip=True)) if soup.title else 0,
      "has_title":int(soup.title is not None), "anchor_count":len(anchors),
      "external_anchor_count":len(ext_a), "anchor_external_ratio":len(ext_a)/max(1,len(anchors)),
      "unique_anchor_hosts":len(set(a_hosts)), "forms_count":len(forms),
      "password_input_count":len(passwords), "input_count":len(inputs), "hidden_input_count":len(hidden),
      "iframe_count":len(soup.find_all("iframe")), "script_count":len(scripts),
      "external_script_count":external(script_hosts), "image_count":len(images),
      "external_image_count":external(image_hosts),
      "javascript_href_count":sum((a.get("href") or "").lower().startswith("javascript:") for a in anchors),
      "mailto_href_count":sum((a.get("href") or "").lower().startswith("mailto:") for a in anchors),
      "forms_without_action":sum(not (f.get("action") or "").strip() for f in forms),
      "form_external_action_count":external(action_hosts),
      "form_http_action_count":sum(urlparse(x).scheme=="http" for x in actions),
      "form_https_action_count":sum(urlparse(x).scheme=="https" for x in actions),
      "forms_with_password":sum(f.find("input",attrs={"type":re.compile("^password$",re.I)}) is not None for f in forms),
      "external_form_action_ratio":external(action_hosts)/max(1,len(forms)),
      "meta_refresh_count":len(soup.find_all("meta",attrs={"http-equiv":re.compile("^refresh$",re.I)})),
      "button_count":len(soup.find_all("button")),
      "email_input_count":sum("email" in (x.get("type","")+" "+x.get("name","")+" "+x.get("id","")).lower() for x in inputs),
      "form_count_with_get":sum((f.get("method") or "get").lower()=="get" for f in forms),
      "form_count_with_post":sum((f.get("method") or "get").lower()=="post" for f in forms),
      "page_url_is_https":int(urlparse(page_url).scheme=="https"), "page_host_length":len(ph),
      "page_host_looks_like_ip":int(bool(re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}",ph)))
    }
    for tag in soup(["script","style","noscript","svg"]): tag.decompose()
    text=re.sub(r"\s+"," ",soup.get_text(" ",strip=True))
    return f,text
