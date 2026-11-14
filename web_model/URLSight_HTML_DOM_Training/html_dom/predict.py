import argparse,json,joblib
from pathlib import Path
from .features import extract_dom_features

def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",required=True); p.add_argument("--html",required=True); p.add_argument("--url",default="")
    a=p.parse_args(); raw=Path(a.html).read_text(encoding="utf-8",errors="replace")
    dom,text=extract_dom_features(raw,a.url); model=joblib.load(a.model)
    rec={"dom_features":dom,"html_text":text}; pred=int(model.predict([rec])[0])
    out={"prediction":"phishing" if pred==1 else "legitimate","label":pred,"url":a.url,
         "warning":"A model prediction is not a guarantee of safety."}
    if hasattr(model,"predict_proba"):
        probs=model.predict_proba([rec])[0]
        out["class_probabilities"]={"legitimate":float(probs[0]),"phishing":float(probs[1])}
    print(json.dumps(out,indent=2))
if __name__=="__main__": main()
