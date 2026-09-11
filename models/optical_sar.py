from PIL import Image,ImageDraw
import numpy as np

def stats(img):
    a=np.asarray(img.convert("RGB")).astype(np.float32)/255
    return {"mean":float(a.mean()),"std":float(a.std())}

def analyze_optical_sar(optical,sar,query):
    o=stats(optical); s=stats(sar)
    w=max(optical.width,sar.width); h=max(optical.height,sar.height)
    c=Image.new("RGB",(2*w,h)); c.paste(optical.resize((w,h)),(0,0)); c.paste(sar.resize((w,h)),(w,0))
    d=ImageDraw.Draw(c); d.text((10,10),"OPTICAL",fill="red"); d.text((w+10,10),"SAR",fill="red")
    return {"answer":f"Paired multisensor evidence processed. Optical mean={o['mean']:.2f}; SAR mean={s['mean']:.2f}.",
            "confidence":"Low — fusion baseline","evidence":c,
            "consistency":{"optical_mean":round(o["mean"],3),"sar_mean":round(s["mean"],3),
                           "note":"Numeric similarity is not semantic agreement."},
            "trace":["Processed optical observation","Processed SAR observation","Combined evidence"],
            "limitations":["True optical/SAR fusion requires modality-aware models and co-registration."]}
