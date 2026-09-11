def route_query(query,img1,img2,meta1,meta2):
    q=query.lower()
    assessment=["suitable","suitability","developed","development","airstrip","infrastructure development","feasible"]
    activity=["movement","activity","traffic","vehicles","activity changes","increased activity","decreased activity"]
    if any(w in q for w in assessment):
        task="suitability_assessment"
        tools=["terrain_evidence","land_cover_evidence","access_evidence","change_evidence","assessment_engine"]
    elif img2 is not None and any(w in q for w in activity):
        task="activity_change"
        tools=["change_detection_baseline","observable_activity_estimator"]
    elif img2 is not None and any(w in q for w in ["change","changed","difference","increase","decrease","between","before","after"]):
        task="change_analysis"
        tools=["change_detection_baseline","change_interpreter"]
    elif img2 is not None:
        task="optical_sar_analysis"
        tools=["optical_evidence","sar_evidence","multisensor_fusion"]
    elif any(w in q for w in ["highlight","where","locate","find","region"]):
        task="grounding"; tools=["text_region_grounding_baseline"]
    elif any(w in q for w in ["describe","caption","scene"]):
        task="caption"; tools=["caption_baseline"]
    else:
        task="vqa"; tools=["vqa_baseline"]
    return {"task":task,"tools":tools,
            "input_configuration":"paired" if img2 is not None else "single_image",
            "parameters":{}}
