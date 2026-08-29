import streamlit as st
import cv2
import time
from moca_engine import MOCAEngine

# Konfigurasi Layar Web
st.set_page_config(page_title="MOCA Dashboard", layout="wide")

# Inisialisasi Memori Interaktif (Session State)
if 'logs' not in st.session_state: st.session_state.logs = []
if 'hadir' not in st.session_state: st.session_state.hadir = 0
if 'ditolak' not in st.session_state: st.session_state.ditolak = 0
if 'last_status' not in st.session_state: st.session_state.last_status = None

# Muat Mesin Utama MOCA
@st.cache_resource
def load_engine():
    return MOCAEngine()
engine = load_engine()

# Injeksi CSS Khusus Dashboard
st.markdown("""
<style>
    .log-card { padding: 16px; border-bottom: 1px solid #CBCBCB; display: flex; align-items: center; justify-content: space-between; }
    .status-dot { width: 28px; height: 28px; border-radius: 50%; margin-right: 16px; }
    .text-title { color: #191C1E; font-size: 14px; font-weight: 600; font-family: Inter; }
    .text-desc { color: #434655; font-size: 12px; font-family: Inter; }
    .time-badge { background: #EDEEF0; padding: 4px 8px; border-radius: 6px; font-size: 10px; color: #737686; }
    .metric-card { text-align: center; padding: 20px; border: 1px solid #CBCBCB; border-radius: 8px; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("Smart Dashboard MOCA: Monitoring K3")

# Tata Letak UI
col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown("<div class='text-title' style='margin-bottom:10px;'>🔴 Live Monitoring</div>", unsafe_allow_html=True)
    FRAME_WINDOW = st.image([])
    
    stat1, stat2 = st.columns(2)
    metric_ditolak = stat1.empty()
    metric_hadir = stat2.empty()

with col2:
    st.markdown("""
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span class='text-title'>Log Pengunjung Lab</span>
            <span style='color:#0E96F1; font-size:12px; font-weight:600;'>Lihat Semua Log</span>
        </div>
        <hr style="margin-top:10px; margin-bottom:0px;">
    """, unsafe_allow_html=True)
    LOG_WINDOW = st.empty()

# Setup Kamera
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

frame_count = 0
current_names = []

# Main Loop Edge Processing
while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame_count += 1
    
    # 1. Eksekusi Face Recognition (Setiap 30 Frame / ~1 detik)
    if frame_count % 30 == 0:
        current_names, st.session_state.hadir = engine.check_attendance(frame)
        
    # 2. Eksekusi YOLO & FSM (Setiap 3 Frame)
    if frame_count % 3 == 0:
        
        annotated_frame, detected_classes = engine.get_detections(frame)
        color, title, desc, waktu = engine.process_fsm(detected_classes, current_names)
        
        engine.update_led(color)
        
        # Mencegah duplikasi log beruntun
        if title != st.session_state.last_status:
            st.session_state.logs.insert(0, f"""
                <div class="log-card">
                    <div style="display:flex; align-items:center;">
                        <div class="status-dot" style="background:{color};"></div>
                        <div>
                            <div class="text-title">{title}</div>
                            <div class="text-desc">{desc}</div>
                        </div>
                    </div>
                    <div class="time-badge">{waktu}</div>
                </div>
            """)
            
            st.session_state.logs = st.session_state.logs[:5] # Batasi 5 log terbaru
            st.session_state.last_status = title
            
            if color == "#CF2C30": 
                st.session_state.ditolak += 1

        # 3. Render Visual UI
        frame_rgb = cv2.cvtColor(cv2.resize(annotated_frame, (480, 360)), cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)
        
        metric_ditolak.markdown(f"<div class='metric-card'><div class='text-title'>Total akses ditolak</div><h1 style='margin:0;'>{st.session_state.ditolak}</h1></div>", unsafe_allow_html=True)
        metric_hadir.markdown(f"<div class='metric-card'><div class='text-title'>Anggota hadir hari ini</div><h1 style='margin:0;'>{st.session_state.hadir}</h1></div>", unsafe_allow_html=True)
        LOG_WINDOW.markdown("".join(st.session_state.logs), unsafe_allow_html=True)
        
    # Jeda pencegah lag WebSocket jaringan
    time.sleep(0.05)