import io
import numpy as np
from PIL import Image
try:
    import rasterio
except Exception:
    rasterio=None

def stretch(arr):
    arr=np.asarray(arr,dtype=np.float32)
    if arr.ndim==3:
        bands=[]
        for band in arr[:3]:
            lo,hi=np.nanpercentile(band,[2,98])
            if hi<=lo: hi=lo+1e-6
            bands.append((np.clip((band-lo)/(hi-lo),0,1)*255).astype(np.uint8))
        while len(bands)<3: bands.append(bands[-1])
        return np.stack(bands[:3],axis=-1)
    lo,hi=np.nanpercentile(arr,[2,98])
    if hi<=lo: hi=lo+1e-6
    return (np.clip((arr-lo)/(hi-lo),0,1)*255).astype(np.uint8)

def load_uploaded_image(uploaded_file):
    data=uploaded_file.read()
    name=uploaded_file.name.lower()
    if name.endswith((".tif",".tiff")):
        if rasterio is None: raise RuntimeError("Install rasterio for GeoTIFF/TIFF.")
        with rasterio.MemoryFile(data) as mem:
            with mem.open() as src:
                arr=src.read()
                meta={"driver":src.driver,"width":src.width,"height":src.height,
                      "count":src.count,"crs":str(src.crs) if src.crs else None,
                      "bounds":list(src.bounds),"transform":list(src.transform),
                      "tags":src.tags()}
        return Image.fromarray(stretch(arr)),meta
    im=Image.open(io.BytesIO(data)).convert("RGB")
    return im,{"driver":"PIL","width":im.width,"height":im.height,"count":3,"crs":None,"bounds":None}

def validate_inputs(img1, img2, meta1, meta2):
    if img1 is None:
        return False, "Image 1 is required."

    if img2 is None:
        return True, "Single image accepted."

    # For the MVP, normalize small dimension differences.
    if img1.size != img2.size:
        print(f"Resizing Image 2 from {img2.size} to {img1.size}")
        img2 = img2.resize(img1.size)

    return True, "Paired images accepted."
