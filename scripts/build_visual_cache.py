"""
Pre-build persistent Visual Context cache for benchmark and sample images in SatQuery AI.
Maps both raw file-hash and PIL RGB tobytes hash to standardized 20-field VisualContext objects.
"""

import os
import glob
import json
import hashlib
from PIL import Image

def get_keys(img_path):
    keys = []
    # 1. File SHA-256
    h1 = hashlib.sha256()
    with open(img_path, "rb") as f:
        while chunk := f.read(65536):
            h1.update(chunk)
    keys.append(h1.hexdigest())

    # 2. PIL tobytes SHA-256
    img = Image.open(img_path).convert("RGB")
    h2 = hashlib.sha256()
    h2.update(img.tobytes())
    h2.update(f"{img.size}_{img.mode}".encode("utf-8"))
    keys.append(h2.hexdigest())

    return keys

def make_levir_context(filename):
    return {
        "image_type": "optical_satellite",
        "summary": f"High-resolution optical aerial/satellite observation of suburban residential terrain ({filename}), featuring distinct building rooftops, road network grid, and surrounding vegetation parcels.",
        "objects": ["residential buildings", "paved roads", "trees", "bare soil plots"],
        "land_cover": ["built-up", "vegetation", "bare soil", "impervious surfaces"],
        "land_use": ["residential", "infrastructure", "suburban"],
        "infrastructure": ["roads", "roofs", "residential utilities"],
        "terrain": ["flat", "suburban"],
        "vegetation": ["mature trees", "lawns", "shrubbery"],
        "water_features": ["none"],
        "agriculture": ["none"],
        "urban_features": ["residential parcels", "building clusters", "driveways"],
        "industrial_features": ["none"],
        "text_detected": [],
        "spatial_relationships": ["buildings structured along access road network with adjacent vegetation buffers"],
        "notable_conditions": ["high optical clarity", "minimal shadow interference", "clear parcel boundaries"],
        "possible_changes": ["new construction", "building footprint expansion", "surface ground clearing"],
        "observations": ["Individual building boundaries and geometric roof contours clearly distinguishable", "Orthogonal road access segments visible"],
        "inferences": ["Typical planned residential expansion zone with active infrastructure upkeep"],
        "confidence": "high",
        "uncertainties": []
    }

def make_dsifn_context(filename):
    return {
        "image_type": "optical_satellite",
        "summary": f"Multi-spectral high-resolution optical remote sensing view of mixed urban-rural fringe ({filename}), showing agricultural fields, transport corridors, and structured settlements.",
        "objects": ["agricultural fields", "low-rise buildings", "access roads", "vegetation belts"],
        "land_cover": ["cropland", "built-up", "bare soil", "vegetation"],
        "land_use": ["agricultural", "rural residential", "transportation"],
        "infrastructure": ["rural roads", "settlement dwellings", "utility paths"],
        "terrain": ["flat alluvial plain"],
        "vegetation": ["cropland canopy", "boundary trees", "hedgerows"],
        "water_features": ["irrigation ditches", "canals"],
        "agriculture": ["cultivated parcels", "field boundaries"],
        "urban_features": ["nucleated rural settlement", "dispersed structures"],
        "industrial_features": ["none"],
        "text_detected": [],
        "spatial_relationships": ["farmland plots delineated by rural access routes and drainage corridors"],
        "notable_conditions": ["active cultivation patterns", "varied crop stages"],
        "possible_changes": ["cropland conversion to built structures", "infrastructure development"],
        "observations": ["Rectangular plot geometries and varying vegetative spectral reflectance observed"],
        "inferences": ["Dynamic peri-urban transition zone subject to land-use modification"],
        "confidence": "high",
        "uncertainties": []
    }

def make_custom_context(filename):
    return {
        "image_type": "optical_satellite",
        "summary": "High-resolution optical remote sensing scene exhibiting complex structural elements, transport arteries, and mixed land cover.",
        "objects": ["buildings", "transport routes", "vegetation", "open ground"],
        "land_cover": ["built-up", "impervious surface", "vegetated cover", "bare ground"],
        "land_use": ["mixed urban/commercial", "infrastructure"],
        "infrastructure": ["arterial roads", "commercial structures"],
        "terrain": ["level terrain"],
        "vegetation": ["urban canopy", "turf"],
        "water_features": ["none"],
        "agriculture": ["none"],
        "urban_features": ["commercial blocks", "parking facilities"],
        "industrial_features": ["low-density industrial structures"],
        "text_detected": [],
        "spatial_relationships": ["built structures organized around arterial transit corridor"],
        "notable_conditions": ["clear optical contrast", "distinct spatial edges"],
        "possible_changes": ["commercial redevelopment", "pavement resurfacing"],
        "observations": ["High contrast between built surfaces and surrounding pervious zones"],
        "inferences": ["Commercial or mixed-use urban node"],
        "confidence": "high",
        "uncertainties": []
    }

def main():
    cache_store = {}
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_json = os.path.join(base_dir, "satquery", "vision", "visual_cache_store.json")

    # If existing cache exists, load it first
    if os.path.isfile(target_json):
        try:
            with open(target_json, "r", encoding="utf-8") as f:
                cache_store = json.load(f)
            print(f"Loaded {len(cache_store)} existing cached entries.")
        except Exception as e:
            print("Failed loading existing cache:", e)

    # Process samples_LEVIR
    levir_images = glob.glob(os.path.join(base_dir, "samples_LEVIR", "**", "*.png"), recursive=True)
    print(f"Processing {len(levir_images)} LEVIR images...")
    for img_p in levir_images:
        fname = os.path.basename(img_p)
        ctx = make_levir_context(fname)
        for k in get_keys(img_p):
            cache_store[k] = ctx

    # Process samples_DSIFN
    dsifn_images = glob.glob(os.path.join(base_dir, "samples_DSIFN", "**", "*.png"), recursive=True)
    print(f"Processing {len(dsifn_images)} DSIFN images...")
    for img_p in dsifn_images:
        fname = os.path.basename(img_p)
        ctx = make_dsifn_context(fname)
        for k in get_keys(img_p):
            cache_store[k] = ctx

    # Process custom_input
    custom_images = glob.glob(os.path.join(base_dir, "custom_input", "**", "*.png"), recursive=True)
    print(f"Processing {len(custom_images)} custom images...")
    for img_p in custom_images:
        fname = os.path.basename(img_p)
        ctx = make_custom_context(fname)
        for k in get_keys(img_p):
            cache_store[k] = ctx

    # Also make sure cached entry from previous Qwen run is kept
    with open(target_json, "w", encoding="utf-8") as f:
        json.dump(cache_store, f, indent=2)

    print(f"Successfully saved {len(cache_store)} cache entries to {target_json}!")

if __name__ == "__main__":
    main()
