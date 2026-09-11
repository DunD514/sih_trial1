import streamlit as st
from preprocessing.image_io import load_uploaded_image, validate_inputs
from agent.router import route_query
from agent.orchestrator import execute_plan

st.set_page_config(page_title="SatQuery AI", page_icon="🛰️", layout="wide")
st.title("🛰️ SatQuery AI")
st.caption("Agentic remote-sensing evidence and assessment prototype")

with st.sidebar:
    st.header("Inputs")
    f1=st.file_uploader("Image 1",type=["tif","tiff","png","jpg","jpeg"])
    f2=st.file_uploader("Image 2 (optional)",type=["tif","tiff","png","jpg","jpeg"])
    st.divider()
    st.write("Single image → VQA / grounding")
    st.write("Two images → change / multisensor analysis")

query=st.text_area("Natural-language query",
                   placeholder="e.g. What changed between these two observations?")

img1=img2=None
meta1=meta2=None
if f1:
    try: img1,meta1=load_uploaded_image(f1)
    except Exception as e: st.error(f"Image 1 error: {e}")
if f2:
    try: img2,meta2=load_uploaded_image(f2)
    except Exception as e: st.error(f"Image 2 error: {e}")

if img1:
    c1,c2=st.columns(2)
    with c1:
        st.subheader("Image 1"); st.image(img1,use_container_width=True)
    if img2:
        with c2:
            st.subheader("Image 2"); st.image(img2,use_container_width=True)

if st.button("Analyze",type="primary",disabled=(img1 is None or not query.strip())):
    ok,msg=validate_inputs(img1,img2,meta1,meta2)
    if not ok:
        st.error(msg)
    else:
        plan=route_query(query,img1,img2,meta1,meta2)
        with st.status("Executing SatQuery workflow...",expanded=True) as status:
            st.write(f"Task: **{plan['task']}**")
            st.write(f"Tools: **{', '.join(plan['tools'])}**")
            result=execute_plan(plan,query,img1,img2,meta1,meta2)
            status.update(label="Workflow complete",state="complete")

        st.divider()
        a,b=st.columns([1.2,1])
        with a:
            st.subheader("Answer / Assessment")
            st.write(result["answer"])
            st.metric("Confidence",result["confidence"])
            if result.get("structured"):
                st.subheader("Structured findings")
                st.json(result["structured"])
        with b:
            st.subheader("Visual evidence")
            if result.get("evidence") is not None:
                st.image(result["evidence"],use_container_width=True)
        st.divider()
        st.subheader("Evidence consistency")
        st.json(result.get("consistency",{}))
        st.subheader("Execution trace")
        for x in result.get("trace",[]): st.write("✓ "+x)
        if result.get("limitations"):
            st.subheader("Limitations")
            for x in result["limitations"]: st.warning(x)

st.divider()
st.caption("Prototype outputs are evidence-based and uncertainty-aware; they are not a substitute for ground truth or professional engineering assessment.")
