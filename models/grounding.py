from PIL import Image,ImageDraw
import numpy as np

def ground_text(image,query):
    a=np.asarray(image.convert("RGB")).astype(np.float32)
    q=query.lower()
    if "water" in q: score=a[:,:,2]-(a[:,:,0]+a[:,:,1])/2
    elif "vegetation" in q or "green" in q: score=a[:,:,1]-(a[:,:,0]+a[:,:,2])/2
    elif "built" in q or "building" in q:
        g=a.mean(axis=2); score=-np.abs(g-np.median(g))
    else: score=a.mean(axis=2)
    ys,xs=np.where(score>=np.percentile(score,92))
    out=image.convert("RGB").copy(); d=ImageDraw.Draw(out)
    if len(xs)>20:
        d.rectangle([int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())],outline="red",width=5)
        ans=f"Candidate region highlighted for: {query}"; conf="Low — heuristic baseline"
    else: ans="No reliable candidate region was found."; conf="Low"
    return {"answer":ans,"confidence":conf,"evidence":out,"consistency":{},
            "trace":["Classified query as grounding","Generated heuristic candidate region"],
            "limitations":["Replace with a validated remote-sensing text-grounding/segmentation model."]}
