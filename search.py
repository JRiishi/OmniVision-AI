import cv2
import os
import numpy as np
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import faiss

# Resolve OpenMP runtime conflicts on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def run_search_engine(video_path, metadata_csv, embeddings_index, output_dir=None):
    if output_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, "output", "search_results")
    
    # Create output dir for matches
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load database files
    print("Loading search database...")
    if not os.path.exists(metadata_csv) or not os.path.exists(embeddings_index):
        print(f"Error: Database files not found. Run indexing first.")
        print(f"Expected metadata CSV: {metadata_csv}")
        print(f"Expected FAISS index: {embeddings_index}")
        return
        
    df = pd.read_csv(metadata_csv)
    index = faiss.read_index(embeddings_index)
    
    print(f"Database loaded: {len(df)} objects indexed in FAISS.")
    
    # 2. Load OpenAI CLIP Model
    print("Loading OpenAI CLIP model (openai/clip-vit-base-patch32)...")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    print("\n=======================================================")
    print("FAISS + CLIP Visual Search Engine is ready!")
    print("Type your search query (e.g. 'a person with red shirt').")
    print("Type 'q' or 'exit' to quit.")
    print("=======================================================\n")
    
    while True:
        query = input("Search query: ").strip()
        if not query or query.lower() in ['q', 'exit', 'quit']:
            print("Exiting search engine.")
            break
            
        print(f"Searching for: '{query}'...")
        
        # 3. Generate CLIP Text Embedding
        try:
            inputs = clip_processor(text=[query], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                text_features = clip_model.get_text_features(**inputs).pooler_output
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                query_embedding = text_features.cpu().numpy().flatten().astype(np.float32)
        except Exception as e:
            print(f"Error generating text embedding: {e}")
            continue
            
        # 4. Search in FAISS index (Inner Product search)
        # Cosine similarity is the inner product of normalized vectors
        k = min(5, len(df))
        similarities, indices = index.search(query_embedding.reshape(1, -1), k)
        
        print("\nTop Matches found:")
        print("---------------------------------------------------------------------------------------------------")
        print(f"{'Rank':<5} | {'Score':<6} | {'Frame':<6} | {'Model':<12} | {'Class':<12} | {'Detected Color':<15} | {'Crop Path'}")
        print("---------------------------------------------------------------------------------------------------")
        
        cap = cv2.VideoCapture(video_path)
        
        for rank, (score, idx) in enumerate(zip(similarities[0], indices[0]), 1):
            if idx == -1:
                continue
            row = df.iloc[idx]
            frame_id = int(row['frame_id'])
            model_name = str(row['model'])
            class_name = str(row['class_name'])
            confidence = float(row['confidence'])
            detected_color = str(row['detected_color'])
            crop_path = str(row['crop_path'])
            bbox = eval(str(row['bbox'])) # convert string to list
            
            print(f"{rank:<5} | {score:.4f} | {frame_id:<6} | {model_name:<12} | {class_name:<12} | {detected_color:<15} | {crop_path}")
            
            # 5. Extract frame and draw bounding box
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if ret:
                x1, y1, x2, y2 = bbox
                # Draw box around object
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
                # Add text label
                label = f"{rank}. Similarity: {score:.2f} ({class_name}, {detected_color})"
                cv2.rectangle(frame, (x1, y1 - 35), (x2, y1), (0, 0, 255), -1)
                cv2.putText(frame, label, (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Save visual match frame
                match_filename = f"match_{rank}_frame_{frame_id}_score_{score:.3f}.jpg"
                match_path = os.path.join(output_dir, match_filename)
                cv2.imwrite(match_path, frame)
                
                # Display match frame if window GUI is available
                try:
                    cv2.imshow(f"Match {rank} (Frame {frame_id})", cv2.resize(frame, (1280, 720)))
                    cv2.waitKey(2000) # Show for 2 seconds
                    cv2.destroyWindow(f"Match {rank} (Frame {frame_id})")
                except Exception:
                    pass
            else:
                print(f"Warning: Could not retrieve frame {frame_id} from video.")
                
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
            
        print("---------------------------------------------------------------------------------------------------")
        print(f"Results saved in: {output_dir}/")
        print("===================================================================================================\n")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(base_dir, "People in the park - 4K Slow motion stock video - 4K Stock Footage Pro (1080p, h264).mp4")
    metadata_csv = os.path.join(base_dir, "output", "video_metadata.csv")
    embeddings_index = os.path.join(base_dir, "output", "embeddings.index")
    
    run_search_engine(
        video_path=video_path,
        metadata_csv=metadata_csv,
        embeddings_index=embeddings_index
    )
