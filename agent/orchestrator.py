from models.vqa import answer_vqa
from models.grounding import ground_text
from models.change_baseline import analyze_change,estimate_activity_change
from models.optical_sar import analyze_optical_sar
from analysis.suitability import assess_suitability

def execute_plan(plan,query,img1,img2,meta1,meta2):
    t=plan["task"]
    if t in ("vqa","caption"): return answer_vqa(img1,query)
    if t=="grounding": return ground_text(img1,query)
    if t=="change_analysis": return analyze_change(img1,img2,query)
    if t=="activity_change": return estimate_activity_change(img1,img2,query)
    if t=="optical_sar_analysis": return analyze_optical_sar(img1,img2,query)
    if t=="suitability_assessment": return assess_suitability(img1,img2,query)
    raise ValueError(t)
