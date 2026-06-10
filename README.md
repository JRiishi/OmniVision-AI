# 🎥 OmniVision AI: Semantic Video Search & Analytics Engine

**OmniVision AI** is a premium, high-performance visual search intelligence platform designed to ingest video footage, detect persons, clothing, and safety gear, classify dominant clothing colors, extract high-dimensional semantic visual embeddings, and enable instant natural language search (e.g., *"a person with red shirt"* or *"someone wearing high-visibility orange jacket"*) with real-time video seeking, highlighting, and relational data analytics.

Designed as an end-to-end computer vision solution, the codebase features a complete **Streamlit Dashboard** and CLI backend using **YOLOv8**, **OpenCV (HSV Segmentation)**, **OpenAI CLIP (Vision Transformer)**, and **FAISS (Facebook AI Similarity Search)**.

---

## 🚀 Key Features

*   **Dual YOLOv8 Model Pipeline**:
    *   **Person & Safety Gear Detector (`best_v2.pt`)**: Extracts persons and high-visibility clothing elements.
    *   **Clothing Categories Detector (`cloth_YOLO.pt`)**: Detects specific articles of clothing (shirts, pants, jackets, skirts, shorts, hats, bags, shoes, sunglasses).
*   **OpenCV HSV dominant Color Extraction**: High-fidelity HSV range masking to identify 11 distinct color channels (`black`, `white`, `grey`, `red`, `orange`, `yellow`, `green`, `blue`, `purple`, `pink`, `brown`) from cropped clothing elements.
*   **OpenAI CLIP Feature Encoding**: Converts cropped visual detections into dense 512-dimensional vector representations using the `openai/clip-vit-base-patch32` visual transformer model.
*   **FAISS Vector Search**: Integrates Facebook AI Similarity Search (`faiss.IndexFlatIP`) utilizing L2-normalized vector representations to perform mathematically exact Cosine Similarity queries at scale.
*   **Modern Web Dashboard**: Implemented in Streamlit using a custom, high-end dark-themed interface with **royal blue-to-teal-to-cyan** gradients, glassmorphism UI components, real-time progress indexing widgets, metadata dataframes, and Plotly/Matplotlib analytics distribution charts.
*   **CLI Toolkit**: Fully-featured terminal tools for index compilation and real-time interactive query rendering.

---

## 🛠️ Tech Stack & Model Details

*   **Computer Vision**: YOLOv8 (Ultralytics) for visual object detection.
*   **Feature Extraction**: OpenAI CLIP (Vision-Language pre-trained Transformer) for cross-modal embedding mapping.
*   **Vector Database**: FAISS (IndexFlatIP) for nearest neighbor cosine-similarity indexing.
*   **Color Classification**: OpenCV (Hue/Saturation/Value segmentations).
*   **Web Framework**: Streamlit (with custom CSS injection).
*   **Data Analysis**: Pandas (relational catalog mapping) & Streamlit charts.
*   **Hardware Acceleration**: PyTorch with automatic platform detection (`cuda` for NVIDIA GPU, `mps` for Apple Silicon GPU, or `cpu` fallback).

---

## 📁 Directory Architecture

```text
OmniVision-AI/
├── app.py                      # Streamlit Search & Analytics Web App (Main UI)
├── main.py                     # Core Video Ingestion, YOLO Detection, & CLIP Indexing Pipeline
├── search.py                   # Interactive Command-line Query Utility (CLI)
├── Architecture.md             # In-depth architectural layout & flow diagrams
├── README.md                   # System configuration and running guide (This file)
├── requirements.txt            # Python dependencies lists
├── best_v2.pt                  # YOLOv8 weights (Person & safety gear)
├── cloth_YOLO.pt               # YOLOv8 weights (Clothing classifications)
└── output/                     # Generated databases and image caches
    ├── active_video.mp4        # Cached uploaded video recording
    ├── video_metadata.csv      # Indexed CSV catalog (Relational schema)
    ├── embeddings.index        # Vector database flat file (FAISS index)
    ├── crops/                  # Directory storing cropped image assets (JPEGs)
    └── search_results/         # Extracted matching frame highlights
```

---

## 📥 Installation & Environment Setup

Follow these steps to set up OmniVision AI on your local system:

### 1. Prerequisites
*   Python 3.10 or higher.
*   `git` installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/JRiishi/OmniVision-AI.git
cd OmniVision-AI
```

### 3. Create a Virtual Environment
It is highly recommended to isolate your project dependencies:
*   **macOS / Linux**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
*   **Windows**:
    ```cmd
    python -m venv .venv
    .venv\Scripts\activate
    ```

### 4. Install Dependencies
Upgrade pip and install the required libraries:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Add Model Weights
Ensure that the two YOLO models are present in the root folder:
*   `best_v2.pt` (Person/high-vis gear detector)
*   `cloth_YOLO.pt` (Clothing classifier)

*Note: Large model files (`.pt`), raw video files, and database cache folders are configured in `.gitignore` to keep git repositories clean.*

---

## 🖥️ Launching the Application

### A. Running the Streamlit Web UI
The easiest and most presentable way to interact with OmniVision AI is through the Streamlit web dashboard. Run:
```bash
streamlit run app.py
```
This launches a local web server (typically on `http://localhost:8501`). The dashboard contains three main tabs:
1.  **Video Ingest & Index**: Upload a video, specify settings (frame skip rate, confidence threshold), and compile the FAISS index database in real-time.
2.  **Semantic Search Interface**: Search for visual elements in natural language (e.g. *"a person wearing a yellow helmet"*, *"blue pants"*). Click the checkbox to view the full matching video frame with overlaid red bounding boxes and classification details.
3.  **Database Analytics**: Visualize summary statistics of your indexed dataset, object class frequencies, and clothing color distributions.

### B. Indexing & Querying via CLI
If you prefer using the command line:

1.  **Ingest and Index Video**:
    Open `main.py`, customize your target video path in the `if __name__ == "__main__":` block, and execute:
    ```bash
    python main.py
    ```
    This prints extraction progress to the terminal and outputs the metadata database (`output/video_metadata.csv`) and FAISS index (`output/embeddings.index`).

2.  **Search the Index**:
    Run the command line query search:
    ```bash
    python search.py
    ```
    Type your description query (e.g., `"a person with red shirt"`) and the script will look up the vector database, write the top matching frame crops to console logs, and export high-resolution matching frame visualizations under `output/search_results/`.

---

## ⚙️ Technical Highlights & Optimization

### 1. Resolving OpenMP Conflicts on macOS
Under macOS, importing FAISS and PyTorch simultaneously can trigger a runtime conflict with the OpenMP implementation (`libomp.dylib`). OmniVision AI handles this automatically by setting:
```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

### 2. Hugging Face Transformers v5 Compatibility
Modern versions of Hugging Face `transformers` wrap CLIP outputs inside structured `BaseModelOutputWithPooling` objects rather than raw PyTorch tensors. Feature extraction in `main.py` is safely configured using:
```python
with torch.no_grad():
    # Retrieve the pooled features output from CLIP
    features = clip_model.get_image_features(**inputs).pooler_output
```
This guarantees robust performance regardless of minor library updates.

### 3. GPU Acceleration
The models automatically check for the best hardware backend. In order of priority:
1.  `cuda` for NVIDIA GPUs.
2.  `mps` (Metal Performance Shaders) for Apple Silicon M1/M2/M3 chips.
3.  `cpu` as a default fallback.

---

## 📊 Relational Database Schema
When a video is indexed, its relational catalog is saved to `output/video_metadata.csv`. The column schema is:
*   `frame_id`: Frame index number from the source video.
*   `model`: The model that detected the object (`best_v2` or `cloth_YOLO`).
*   `object_id`: Index of the detected object in that frame.
*   `class_name`: Detected label (e.g., `person`, `jacket`, `hat`, etc.).
*   `confidence`: Object detector confidence score (0 to 1).
*   `crop_path`: The disk path to the saved cropped image JPEG.
*   `bbox`: Coordinates of the bounding box `[x1, y1, x2, y2]`.
*   `detected_color`: Color classification from OpenCV (or `N/A` for general person bounding boxes).

---

## 📜 License
This project is prepared as a custom visual search application interface. All rights reserved.
