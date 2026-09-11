from functools import lru_cache

@lru_cache(maxsize=1)
def load_blip():
    from transformers import BlipProcessor,BlipForQuestionAnswering
    import torch
    p=BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    m=BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    m.eval()
    return p,m

def answer_vqa(image,question):
    try:
        import torch
        p,m=load_blip()
        x=p(images=image,text=question,return_tensors="pt")
        with torch.no_grad(): y=m.generate(**x,max_new_tokens=40)
        ans=p.decode(y[0],skip_special_tokens=True)
        return {"answer":ans,"confidence":"Uncalibrated baseline","evidence":image,
                "consistency":{},"trace":["Loaded VQA baseline","Processed image and question","Generated answer"],
                "limitations":["BLIP is a generic baseline; replace it with a remote-sensing-adapted VLM for final evaluation."]}
    except Exception as e:
        return {"answer":f"VQA could not run: {e}","confidence":"Unavailable","evidence":image,
                "consistency":{},"trace":["VQA execution attempted"],
                "limitations":["Fix the Hugging Face/PyTorch environment before running VQA."]}
