import argparse, json
from .scraper import scrape
def main():
    p=argparse.ArgumentParser(); p.add_argument("url"); p.add_argument("--name",default="page"); p.add_argument("--output",default="webscraper_output")
    a=p.parse_args()
    try:
        r=scrape(a.url,f"{a.output}/{a.name}"); r["success"]=True
    except Exception as e:
        r={"requested_url":a.url,"success":False,"error":str(e)}
    print(json.dumps(r,indent=2))
if __name__=="__main__": main()
