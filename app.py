import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
import torch
import faiss
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from main import process_video_pipeline, get_dominant_color

# Resolve OpenMP runtime conflicts on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 1. Set Page Configuration
st.set_page_config(
    page_title="OmniVision AI - Visual Search & Analytics",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Premium CSS Custom Theme Injection
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background & Styling */
    .stApp {
        background-color: #0f111a;
        color: #e2e8f0;
    }
    
    /* Header Gradient */
    .main-title {
        background: linear-gradient(135deg, #2563eb 0%, #0d9488 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
        letter-spacing: -0.05rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Glassmorphism card container */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(59, 130, 246, 0.5);
        transform: translateY(-2px);
    }
    
    /* Custom Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 20px;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .badge-primary { background-color: rgba(59, 130, 246, 0.25); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }
    .badge-secondary { background-color: rgba(13, 148, 136, 0.25); color: #2dd4bf; border: 1px solid rgba(13, 148, 136, 0.4); }
    .badge-success { background-color: rgba(16, 185, 129, 0.25); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .badge-color { background-color: rgba(245, 158, 11, 0.25); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
    
    /* Metric styling */
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #3b82f6;
        margin-bottom: 0px;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05rem;
    }
    
    /* Input border customization */
    .stTextInput>div>div>input {
        background-color: #1e293b;
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }
    
    /* Button custom hover styling */
    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #0d9488 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #0f766e 100%);
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Global paths config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_1 = os.path.join(BASE_DIR, "best_v2.pt")
YOLO_MODEL_2 = os.path.join(BASE_DIR, "cloth_YOLO.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
METADATA_CSV = os.path.join(OUTPUT_DIR, "video_metadata.csv")
FAISS_INDEX = os.path.join(OUTPUT_DIR, "embeddings.index")
VIDEO_SAVE_PATH = os.path.join(OUTPUT_DIR, "active_video.mp4")

# Load models helper (cached to avoid reloading on every interaction)
@st.cache_resource
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor, device

# App Sidebar
st.sidebar.markdown("<h2 style='color:#3b82f6;'>⚙️ Configuration</h2>", unsafe_allow_html=True)
frame_skip_rate = st.sidebar.slider("Frame Skip Rate", min_value=1, max_value=120, value=30, step=5,
                                    help="Process every Nth frame. Higher = faster indexing, Lower = thorough scan.")
conf_threshold = st.sidebar.slider("Detection Confidence", min_value=0.1, max_value=0.9, value=0.15, step=0.05,
                                   help="Confidence threshold for YOLO predictions.")

st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style='font-size:0.8rem; color:#64748b;'>
        <b>OmniVision AI Engine v1.0</b><br>
        Internship Project Submission<br>
        Integrated Stack:<br>
        - YOLOv8 (Dual Models)<br>
        - OpenCV HSV Segmenter<br>
        - OpenAI CLIP Encoder<br>
        - FAISS Vector Database
    </div>
    """, unsafe_allow_html=True)

# Header layout
st.markdown("<h1 class='main-title'>🎥 OmniVision AI</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Intelligent Video Semantic Search & Vector Analytics Engine</div>", unsafe_allow_html=True)

# Define Tabs
tab1, tab2, tab3 = st.tabs(["🎥 Ingest & Index Video", "🔍 Semantic Search Engine", "📊 Database Analytics"])

# Tab 1: Ingest & Index Video
with tab1:
    st.subheader("Upload and Index Video Recording")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload Video Recording (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(VIDEO_SAVE_PATH, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("Video successfully uploaded and saved!")
            st.video(VIDEO_SAVE_PATH)
            
    with col2:
        if uploaded_file is not None:
            st.markdown("### Process Indexing Pipeline")
            st.write("Ready to index video. This will perform dual YOLO object detection, classify clothing colors with OpenCV, extract semantic visual features with CLIP, and compile a vector search database.")
            
            if st.button("⚡ Start AI Indexing"):
                status_placeholder = st.empty()
                progress_bar = st.progress(0.0)
                logs_area = st.empty()
                
                # Callback to update progress in UI
                def ui_progress_callback(frame_count, total_frames):
                    prog = min(1.0, float(frame_count) / float(total_frames))
                    progress_bar.progress(prog)
                    status_placeholder.info(f"Processing frame: {frame_count} / {total_frames} ({int(prog * 100)}%)")
                
                with st.spinner("Processing... Running deep learning models..."):
                    process_video_pipeline(
                        video_path=VIDEO_SAVE_PATH,
                        yolo_model_path_1=YOLO_MODEL_1,
                        yolo_model_path_2=YOLO_MODEL_2,
                        output_dir=OUTPUT_DIR,
                        frame_skip=frame_skip_rate,
                        progress_callback=ui_progress_callback
                    )
                
                progress_bar.progress(1.0)
                status_placeholder.success("🎉 Video Indexing Pipeline Completed Successfully!")
                
                # Load metadata summary
                if os.path.exists(METADATA_CSV):
                    df = pd.read_csv(METADATA_CSV)
                    st.markdown("---")
                    st.markdown("### Indexing Results")
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.markdown(f"<div class='glass-card'><div class='metric-value'>{len(df)}</div><div class='metric-label'>Objects Detected</div></div>", unsafe_allow_html=True)
                    with col_m2:
                        st.markdown(f"<div class='glass-card'><div class='metric-value'>{df['frame_id'].nunique()}</div><div class='metric-label'>Frames Indexed</div></div>", unsafe_allow_html=True)
                    with col_m3:
                        st.markdown(f"<div class='glass-card'><div class='metric-value'>{df['class_name'].nunique()}</div><div class='metric-label'>Unique Classes</div></div>", unsafe_allow_html=True)
        else:
            st.info("Upload a video recording in the uploader pane to start.")

# Tab 2: Semantic Search Engine
with tab2:
    st.subheader("Semantic Query Vector Search")
    
    # Check if files exist
    if not os.path.exists(METADATA_CSV) or not os.path.exists(FAISS_INDEX):
        st.warning("⚠️ Database files not found. Please upload and index a video in Tab 1 first.")
    else:
        # Load DB in memory
        df_db = pd.read_csv(METADATA_CSV)
        faiss_index_db = faiss.read_index(FAISS_INDEX)
        clip_model, clip_processor, clip_device = load_clip_model()
        
        # Search Bar - Unified layout
        col_text, col_btn = st.columns([7, 1], vertical_alignment="bottom")
        
        with col_text:
            query_text = st.text_input(
                "Enter natural language description:", 
                placeholder="e.g. a person with red shirt, white jacket, hat, etc.",
                key="search_query"
            )
            
        with col_btn:
            run_search = st.button("🔍 Search", use_container_width=True)
            
        if (run_search or query_text) and query_text:
            st.markdown(f"**Search Results for:** *\"{query_text}\"*")
            
            # 1. Encode text query to CLIP embedding
            with torch.no_grad():
                inputs = clip_processor(text=[query_text], return_tensors="pt", padding=True).to(clip_device)
                text_features = clip_model.get_text_features(**inputs).pooler_output
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                query_embedding = text_features.cpu().numpy().flatten().astype(np.float32)
                
            # 2. Search FAISS index
            k_hits = min(6, len(df_db))
            similarities, indices = faiss_index_db.search(query_embedding.reshape(1, -1), k_hits)
            
            # Show grid matches
            st.markdown("### Top Matches")
            
            # Setup columns for matches
            cols = st.columns(3)
            
            cap = cv2.VideoCapture(VIDEO_SAVE_PATH)
            
            for rank, (score, idx) in enumerate(zip(similarities[0], indices[0]), 1):
                if idx == -1:
                    continue
                row = df_db.iloc[idx]
                frame_id = int(row['frame_id'])
                model_name = str(row['model'])
                class_name = str(row['class_name'])
                detected_color = str(row['detected_color'])
                crop_path = str(row['crop_path'])
                bbox = eval(str(row['bbox']))
                
                # Get column
                col_ui = cols[(rank - 1) % 3]
                
                with col_ui:
                    # Glass-card style container
                    st.markdown(f"""
                        <div class='glass-card'>
                            <h4 style='margin-top:0px;color:#60a5fa;'>Match #{rank} (Frame {frame_id})</h4>
                            <div style='margin-bottom:10px;'>
                                <span class='badge badge-primary'>Sim: {score:.2f}</span>
                                <span class='badge badge-secondary'>{model_name}</span>
                                <span class='badge badge-success'>{class_name}</span>
                                <span class='badge badge-color'>Color: {detected_color}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display the cropped image
                    if os.path.exists(crop_path):
                        crop_img = Image.open(crop_path)
                        st.image(crop_img, caption="Object Crop", use_container_width=True)
                    else:
                        st.error("Crop image not found")
                        
                    # Option to show full matching frame
                    if st.checkbox("🔍 View Highlighted Frame Match", key=f"match_{rank}_{idx}"):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                        ret, frame = cap.read()
                        if ret:
                            # Draw bbox
                            x1, y1, x2, y2 = bbox
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            
                            # Add text label
                            label = f"{rank}. Similarity: {score:.2f} ({class_name}, {detected_color})"
                            cv2.rectangle(frame, (x1, y1 - 35), (x2, y1), (0, 0, 255), -1)
                            cv2.putText(frame, label, (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                            
                            # Convert to RGB and show
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            st.image(rgb_frame, caption=f"Matched Frame {frame_id}", use_container_width=True)
                        else:
                            st.warning("Could not extract frame from video")
            cap.release()

# Tab 3: Database Analytics
with tab3:
    st.subheader("Database Vector Metrics & Analytics")
    
    if not os.path.exists(METADATA_CSV):
        st.warning("⚠️ Database metadata not found. Please upload and index a video first.")
    else:
        df_analytics = pd.read_csv(METADATA_CSV)
        
        # Key Summary Metrics
        st.markdown("### Database Overview")
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        
        with col_c1:
            st.metric("Total Indexed Elements", len(df_analytics))
        with col_c2:
            st.metric("Unique Frames Scanning", df_analytics['frame_id'].nunique())
        with col_c3:
            st.metric("Class Categories Detected", df_analytics['class_name'].nunique())
        with col_c4:
            st.metric("Average Confidence Score", f"{df_analytics['confidence'].mean():.2f}")
            
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.markdown("#### Object Class Distribution")
            class_counts = df_analytics['class_name'].value_counts()
            st.bar_chart(class_counts)
            
        with col_g2:
            st.markdown("#### Dominant Clothing Color Distribution")
            # Exclude N/A values which belong to person detections
            clothing_colors = df_analytics[df_analytics['detected_color'] != 'N/A']['detected_color'].value_counts()
            if len(clothing_colors) > 0:
                st.bar_chart(clothing_colors)
            else:
                st.info("No clothing colors detected yet.")
                
        st.markdown("---")
        st.markdown("### Metadata Explorer")
        st.dataframe(df_analytics, use_container_width=True)
