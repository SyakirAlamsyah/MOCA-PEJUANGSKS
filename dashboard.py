import streamlit as st
import cv2
import time
from moca_engine import MOCAEngine

st.set_page_config(page_title="MOCA Dashboard", layout="wide")

if 'logs' not in st.session_state: st.session_state.logs = []
if 'ditolak' not in st.session_state: st.session_state.ditolak = 0
if 'last_status' not in st.session_state: st.session_state.last_status = None

@st.cache_resource
def load_engine(): return MOCAEngine()
engine = load_engine()

# ... (Masukkan blok CSS dan Layout UI Streamlit dari kode sebelumnya di sini) ...
col1, col2 = st.columns([1.5, 1])
with col1:
    FRAME_WINDOW = st.image([])
    stat1, stat2 = st.columns(2)
    metric_ditolak = stat1.empty()
    metric_hadir = stat2.empty()
with col2:
    LOG_WINDOW = st.empty()

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame_count += 1
    if frame_count % 3 != 0: continue
        
    # 1. Eksekusi Face Recognition (Identitas & Total Hadir Unik)
    recognized_names, total_hadir = engine.check_attendance(frame)
    
    # 2. Eksekusi YOLO
    results = engine.model.predict(source=frame, show=False, verbose=False)
    annotated_frame, detected_classes = engine.draw_annotations(frame, results)
    
    # 3. Proses FSM Gabungan (APD + Identitas)
    color, title, desc, waktu = engine.process_fsm(detected_classes, recognized_names)
    
    if title != st.session_state.last_status:
        st.session_state.logs.insert(0, f"""
            <div style="padding: 16px; border-bottom: 1px solid #CBCBCB; display: flex; align-items: center;">
                <div style="width: 28px; height: 28px; border-radius: 50%; margin-right: 16px; background:{color};"></div>
                <div><div style="color: #191C1E; font-weight: 600;">{title}</div><div style="font-size: 12px;">{desc}</div></div>
            </div>
        """)
        st.session_state.logs = st.session_state.logs[:5]
        st.session_state.last_status = title
        if color == "#CF2C30": st.session_state.ditolak += 1

    frame_rgb = cv2.cvtColor(cv2.resize(annotated_frame, (480, 360)), cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame_rgb)
    metric_ditolak.markdown(f"<div style='text-align:center; padding:20px; border:1px solid #CBCBCB;'>Total Ditolak<h1>{st.session_state.ditolak}</h1></div>", unsafe_allow_html=True)
    metric_hadir.markdown(f"<div style='text-align:center; padding:20px; border:1px solid #CBCBCB;'>Anggota Hadir<h1>{total_hadir}</h1></div>", unsafe_allow_html=True)
    LOG_WINDOW.markdown("".join(st.session_state.logs), unsafe_allow_html=True)
    
    time.sleep(0.1)