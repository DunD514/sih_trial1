import numpy as np
from PIL import Image


def _prepare_images(img1, img2):
    """
    Make the two images the same dimensions before numerical comparison.

    For the MVP we resize Image 2 to Image 1's dimensions.
    Later, for real GeoTIFF processing, this should be replaced by
    proper geospatial reprojection/co-registration.
    """

    img1 = img1.convert("RGB")
    img2 = img2.convert("RGB")

    if img1.size != img2.size:
        print(
            f"[Change Detection] "
            f"Resizing Image 2 from {img2.size} "
            f"to {img1.size}"
        )

        img2 = img2.resize(
            img1.size,
            Image.Resampling.BILINEAR
        )

    return img1, img2


def _diff(img1, img2):
    """
    Calculate a simple pixel-level difference.
    """

    img1, img2 = _prepare_images(img1, img2)

    x = np.asarray(img1).astype(np.float32) / 255.0
    y = np.asarray(img2).astype(np.float32) / 255.0

    # Both arrays are now guaranteed to have the same shape.
    difference = np.mean(
        np.abs(x - y),
        axis=2
    )

    return difference


def analyze_change(img1, img2, query):

    # Calculate difference
    difference = _diff(img1, img2)

    # Find pixels with unusually large differences
    threshold = np.percentile(
        difference,
        90
    )

    change_mask = difference >= threshold

    # Percentage of image classified as changed
    changed_fraction = float(
        change_mask.mean()
    )

    # Prepare output image
    _, img2_resized = _prepare_images(
        img1,
        img2
    )

    overlay = np.asarray(
        img2_resized.convert("RGB")
    ).copy()

    # Mark detected change
    overlay[change_mask] = [
        255,
        0,
        0
    ]

    # Simple description
    if changed_fraction < 0.03:

        magnitude = "limited"

    elif changed_fraction < 0.15:

        magnitude = "moderate"

    else:

        magnitude = "substantial"

    answer = (
        f"The baseline detects {magnitude} "
        f"visual change affecting approximately "
        f"{changed_fraction * 100:.1f}% "
        f"of the image area."
    )

    return {

        "answer": answer,

        "confidence": "Threshold-derived baseline",

        "evidence": Image.fromarray(
            overlay
        ),

        "consistency": {},

        "trace": [

            "Received two image observations",

            "Checked image dimensions",

            "Normalized image dimensions",

            "Computed pixel-level temporal difference",

            "Thresholded high-difference pixels",

            "Generated change overlay"

        ],

        "limitations": [

            "This is a pixel-difference baseline.",

            "Pixel differences can be caused by lighting, "
            "clouds, haze, seasonal variation, sensor differences "
            "or image misregistration.",

            "The baseline does not understand semantic changes "
            "such as new buildings or changed infrastructure.",

            "A trained remote-sensing change-detection model "
            "should replace this component for final evaluation."

        ]

    }


def estimate_activity_change(
    img1,
    img2,
    query
):

    difference = _diff(
        img1,
        img2
    )

    threshold = np.percentile(
        difference,
        97
    )

    change_mask = (
        difference >= threshold
    )

    changed_fraction = float(
        change_mask.mean()
    )

    _, img2_resized = _prepare_images(
        img1,
        img2
    )

    overlay = np.asarray(
        img2_resized.convert("RGB")
    ).copy()

    overlay[change_mask] = [
        255,
        0,
        0
    ]

    if changed_fraction < 0.01:

        trend = "little observable difference"

    elif changed_fraction < 0.04:

        trend = "moderate observable change"

    else:

        trend = "substantial observable change"

    answer = (

        f"The available observations show "
        f"{trend}. "

        "This is an image-change indicator, "
        "not a direct count or identification "
        "of vehicles."

    )

    return {

        "answer": answer,

        "confidence": "Low/medium baseline",

        "evidence": Image.fromarray(
            overlay
        ),

        "consistency": {},

        "trace": [

            "Received temporal observations",

            "Normalized image dimensions",

            "Computed temporal difference",

            "Estimated high-change fraction",

            "Generated observable-activity indicator"

        ],

        "limitations": [

            "Two-image pixel difference cannot reliably "
            "establish heavy vehicle movement.",

            "Vehicle activity should eventually be estimated "
            "using a trained object detector across multiple "
            "observations and acquisition times."

        ]

    }