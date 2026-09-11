import argparse,pandas as pd
from datasets import Dataset

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv",required=True)
    p.add_argument("--out",default="prepared_bigearthnet")
    a=p.parse_args()
    df=pd.read_csv(a.csv)
    print("Columns:",df.columns.tolist())
    print("Rows:",len(df))
    img=None; txt=None
    for c in ["image","image_path","path","id","patch_id"]:
        if c in df.columns: img=c; break
    for c in ["text","caption","description","sentences","labels"]:
        if c in df.columns: txt=c; break
    if img is None or txt is None:
        raise ValueError("Set the image/text column names manually after inspecting your exact schema.")
    out=df[[img,txt]].copy(); out.columns=["image","text"]
    out.to_csv(a.out+".csv",index=False)
    Dataset.from_pandas(out).save_to_disk(a.out)
    print("Saved",a.out)

if __name__=="__main__": main()
