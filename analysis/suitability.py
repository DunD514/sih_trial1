import numpy as np
from PIL import Image,ImageDraw

def feat(img):
    a=np.asarray(img.convert("RGB")).astype(np.float32)/255
    g=a.mean(axis=2)
    return {"texture":float(g.std()),
            "open":float((g>np.percentile(g,45)).mean()),
            "water":float(a[:,:,2].mean())}

def assess_suitability(img1,img2,query):
    f=feat(img1)
    terrain=max(0,min(1,1-f["texture"]*2))
    open_score=max(0,min(1,f["open"]))
    water_penalty=max(0,min(.6,f["water"]*.5))
    score=max(0,min(1,.45*terrain+.55*open_score-water_penalty))
    cat="more favorable" if score>=.7 else ("mixed" if score>=.45 else "less favorable")
    ans=(f"Based only on observable imagery, the area is {cat} for the queried development scenario. "
         "This is a screening assessment, not an engineering or operational recommendation.")
    structured={"open_area_proxy":round(open_score,3),"terrain_uniformity_proxy":round(terrain,3),
                "water_constraint_proxy":round(water_penalty,3),"screening_score":round(score,3)}
    e=img1.convert("RGB").copy(); ImageDraw.Draw(e).text((10,10),f"Screening score: {score:.2f}",fill="red")
    return {"answer":ans,"confidence":"Low — transparent screening baseline","evidence":e,
            "structured":structured,"consistency":{},
            "trace":["Extracted image-level terrain/open-area proxies","Computed transparent screening rubric",
                     "Reported uncertainty and limitations"],
            "limitations":["Imagery alone cannot establish soil bearing capacity, engineering feasibility, ownership, logistics, or other non-imagery constraints.",
                           "Replace proxies with validated DEM/land-cover/road/water layers and domain-specific models."]}
