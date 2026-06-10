# OmniVision AI: Semantic Video Search & Analytics Engine

OmniVision AI is an intelligent visual search engine designed to scan video recordings, detect objects of interest, and enable natural language queries (e.g., *"a person with red shirt"*) to locate corresponding matching frames.

It integrates state-of-the-art computer vision models, color extraction heuristics, and vector search indexing into a single, cohesive application.

---

## Key Features

1. **Dual YOLO Object Detection**:
   - Uses YOLOv8 (`best_v2.pt`) to locate and crop persons and safety gear.
   - Uses YOLOv8 (`cloth_YOLO.pt`) to locate and crop clothing items (shirts, pants, jackets, shoes, etc.).
2. **OpenCV Color Classification**:
   - Converts clothing crops to the HSV color space and runs pixel-mask segmentations to automatically determine dominant clothing colors (e.g. `red`, `blue`, `green`, `black`).
3. **OpenAI CLIP Semantic Embeddings**:
   - Converts BGR image crops into 512-dimensional vector representations using the `openai/clip-vit-base-patch32` visual transformer.
   - Normalizes embeddings to unit vectors to perform search queries using Cosine Similarity.
4. **FAISS Vector Database**:
   - Stores and searches high-dimensional embeddings using a flat inner product index (`faiss.IndexFlatIP`), enabling nearest-neighbor search.
5. **Classy Streamlit Interface**:
   - A dark-mode, royal blue-to-teal dashboard providing drag-and-drop video upload, real-time pipeline indexing with progress bars, visual similarity cards showing matches, seek-to-frame highlights, and database analytics charts.

---

## System Workflows

### 1. Ingestion & Database Indexing Pipeline

When a video is indexed:
1. **Frame Selection**: The video is read frame-by-frame, applying a configurable frame skipping rate (e.g., scanner runs on every 30th frame to minimize computation).
2. **Object Detection**:
   - **Model 1 (`best_v2`)** detects people and crops them. Bounding boxes are recorded.
   - **Model 2 (`cloth_YOLO`)** detects clothing items and crops them. Bounding boxes are recorded.
3. **Color & CLIP Feature Extraction**:
   - Clothing crops undergo HSV segmentation inside `get_dominant_color` to classify color names.
   - All cropped images are fed into CLIP (`clip-vit-base-patch32`) to compute a 512-dimensional semantic feature vector.
4. **Database Storage**:
   - Structural details are written to `output/video_metadata.csv`.
   - Embeddings are added to the FAISS index database and written to `output/embeddings.index`.

### 2. Search & Retrieval Pipeline

When a search query (e.g. *"white jacket"*) is typed:
1. **Query Encoding**: The text is converted into a 512-dimensional vector embedding using the CLIP text model.
2. **Vector Search**: The query embedding is passed to `faiss_index.search()` to find the nearest object crops.
3. **Result Presentation**:
   - The UI shows matched cropped cards sorted by similarity scores.
   - Selecting a match seeks to the exact frame in the video file using OpenCV, draws a red bounding box around the matched coordinates, and overlays a label showing the rank, similarity, class, and color.

---

## Directory Organization

```
AI_VIDEO_SEARCH/
├── app.py                      # Streamlit Search & Analytics Web App (UI)
├── main.py                     # Video Ingestion Pipeline & Video Comparison Player
├── search.py                   # Command-line (CLI) search index query script
├── Architecture.md             # Detailed engineering system layout documentation
├── README.md                   # Installation, overview, and running guide (This file)
├── best_v2.pt                  # YOLOv8 Person & Gear detection weights
├── cloth_YOLO.pt               # YOLOv8 Clothing detection weights
└── output/                     # Index databases and crop images folder
    ├── active_video.mp4        # Saved uploaded video recording
    ├── video_metadata.csv      # Indexed metadata CSV relational database
    ├── embeddings.index        # FAISS vector database file
    ├── crops/                  # Cropped JPEGs of all detected objects
    └── search_results/         # Extracted matching search result frames
```

---

## Installation & Setup

Ensure Python 3.10+ and a virtual environment are set up:

1. **Activate the Virtual Environment**:
   ```bash
   source .venv/bin/activate
   ```

2. **Run the Streamlit Dashboard**:
   ```bash
   streamlit run app.py
   ```
   *This will launch the local web server on [http://localhost:8501](http://localhost:8501).*

---

## Pipeline Execution options via Terminal CLI

As a developer, you can also run indexing and queries directly via terminal commands:

- **Run Indexing**:
  Modify parameters in the `__main__` block of `main.py` and execute:
  ```bash
  python main.py
  ```

- **Query Database**:
  Run the CLI search utility:
  ```bash
  python search.py
  ```
  *(Enter your query, e.g., "a person with red shirt", and inspect match logs in the console or JPEGs saved under `output/search_results/`.)*
