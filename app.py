import os
import streamlit as st
from core.pipeline import process_video, VEHICLES
from core.audio import add_alert_sound

st.set_page_config(page_title="Smart Traffic Monitoring", layout="wide")
st.title("🚦 Smart Traffic Monitoring System")

with st.sidebar:
    conf = st.slider("Confidence", 0.1, 0.9, 0.3, 0.05)
    names = st.multiselect("Classes", list(VEHICLES.values()), default=list(VEHICLES.values()))
    model_name = st.selectbox("Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
    line_y = st.slider("Counting line position", 0.1, 0.9, 0.5, 0.05)
    mpp = st.number_input("Meters per pixel (calibration)", value=0.05, format="%.4f")
    skip = st.selectbox("Process every Nth frame", [1, 2, 3])
    heat = st.checkbox("Heatmap overlay", value=False)
    direction = st.selectbox("Normal traffic direction (down = towards bottom of screen)", ["down", "up"])

up = st.file_uploader("Upload traffic video", type=["mp4", "avi", "mov", "mkv"])

if up and st.button("Start Tracking", type="primary"):
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    src = os.path.join("uploads", up.name)
    with open(src, "wb") as f:
        f.write(up.read())
    dst = os.path.join("outputs", "tracked_" + os.path.splitext(up.name)[0] + ".mp4")
    csv_path = dst.replace(".mp4", ".csv")
    class_ids = [k for k, v in VEHICLES.items() if v in names]

    frame_box = st.empty()
    bar = st.progress(0.0)
    c1, c2, c3, c4 = st.columns(4)
    cards = [c1.empty(), c2.empty(), c3.empty(), c4.empty()]

    def on_frame(img, s):
        frame_box.image(img, channels="RGB", use_container_width=True)
        cards[0].metric("In frame", s["current"])
        cards[1].metric("Total", s["total"])
        cards[2].metric("IN", s["in_count"])
        cards[3].metric("OUT", s["out_count"])

    stats = process_video(src, dst, csv_path, conf=conf, classes=class_ids,
                          model_name=model_name, meters_per_pixel=mpp,
                          line_y_ratio=line_y, use_heatmap=heat, frame_skip=skip, expected_dir=direction,
                          on_progress=bar.progress, on_frame=on_frame)
    final = dst.replace(".mp4", "_alert.mp4")
    try:
        add_alert_sound(dst, stats["alert_times"], final)
        dst = final
    except Exception as e:
        st.warning(f"Alert sound add nahi hua: {e}")
    st.success("Done!")
    st.video(dst)
    st.write(f"**Avg speed:** {stats['avg_speed']:.1f} km/h | **Max:** {stats['max_speed']:.1f} km/h (approximate)")
    d1, d2 = st.columns(2)
    d1.download_button("Download video", open(dst, "rb"), file_name=os.path.basename(dst))
    d2.download_button("Download CSV report", open(csv_path, "rb"), file_name=os.path.basename(csv_path))

    st.subheader("Wrong-way vehicles report")
    viol = stats["violations"]
    if not viol:
        st.info("No wrong-way vehicle detected.")
    for x in viol:
        a, b = st.columns([1, 2])
        a.image(x["image"])
        b.markdown(f"**ID #{x['track_id']}** | {x['vehicle']} | time {x['time_sec']}s  \nNumber plate: **{x['plate']}**")
