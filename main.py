import cv2
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import faiss



def get_dominant_color(bgr_image):
    """
    Given a BGR image crop, return the dominant color name.
    """
    if bgr_image is None or bgr_image.size == 0:
        return "Unknown"
        
    # Convert crop to HSV
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    
    # Define standard color ranges in HSV
    # Hue: [0, 180], Saturation: [0, 255], Value: [0, 255]
    color_ranges = {
        "black": [
            (np.array([0, 0, 0]), np.array([180, 255, 50]))
        ],
        "white": [
            (np.array([0, 0, 200]), np.array([180, 40, 255]))
        ],
        "grey": [
            (np.array([0, 0, 50]), np.array([180, 40, 199]))
        ],
        "red": [
            (np.array([0, 50, 50]), np.array([10, 255, 255])),
            (np.array([170, 50, 50]), np.array([180, 255, 255]))
        ],
        "orange": [
            (np.array([11, 50, 50]), np.array([25, 255, 255]))
        ],
        "yellow": [
            (np.array([26, 50, 50]), np.array([35, 255, 255]))
        ],
        "green": [
            (np.array([36, 50, 50]), np.array([85, 255, 255]))
        ],
        "blue": [
            (np.array([86, 50, 50]), np.array([125, 255, 255]))
        ],
        "purple": [
            (np.array([126, 50, 50]), np.array([145, 255, 255]))
        ],
        "pink": [
            (np.array([146, 50, 50]), np.array([169, 255, 255]))
        ],
        "brown": [
            (np.array([10, 50, 30]), np.array([20, 255, 120]))
        ]
    }
    
    color_counts = {}
    for color_name, ranges in color_ranges.items():
        total_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lower, upper) in ranges:
            mask = cv2.inRange(hsv, lower, upper)
            total_mask = cv2.bitwise_or(total_mask, mask)
        color_counts[color_name] = cv2.countNonZero(total_mask)
        
    # Get the color with the maximum count
    max_color = max(color_counts, key=color_counts.get)
    
    # If the maximum count is 0, return unknown
    if color_counts[max_color] == 0:
        return "Unknown"
        
    return max_color

def process_video_pipeline(video_path, 
                           yolo_model_path_1=None, 
                           yolo_model_path_2=None, 
                           output_dir="output", 
                           frame_skip=30,
                           progress_callback=None):
    """
    Pipeline to extract frames, run two YOLO models (best_v2 for persons and cloth_YOLO for clothing items),
    detect clothing colors using OpenCV, generate OpenAI CLIP embeddings for all object crops,
    and save all crop and metadata.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if yolo_model_path_1 is None:
        yolo_model_path_1 = os.path.join(base_dir, "best_v2.pt")
    if yolo_model_path_2 is None:
        yolo_model_path_2 = os.path.join(base_dir, "cloth_YOLO.pt")

    os.makedirs(output_dir, exist_ok=True)
    crops_dir = os.path.join(output_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    
    # 1. Load YOLO models
    print(f"Loading Model 1 (Persons) from: {yolo_model_path_1}")
    model1 = YOLO(yolo_model_path_1, task='detect')
    print(f"Model 1 classes: {model1.names}")
    
    print(f"Loading Model 2 (Clothing) from: {yolo_model_path_2}")
    model2 = YOLO(yolo_model_path_2, task='detect')
    print(f"Model 2 classes: {model2.names}")
    
    # 2. Load OpenAI CLIP Model & Processor
    print("Loading OpenAI CLIP model (openai/clip-vit-base-patch32)...")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device for CLIP: {device}")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # 3. Open Video
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        print("Please provide a valid video file.")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 1
        
    frame_count = 0
    saved_data = [] # To store metadata for pandas dataframe
    embeddings_list = [] # To store CLIP features

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process 1 frame every 'frame_skip' frames to save computation
        if frame_count % frame_skip == 0:
            print(f"Processing frame {frame_count} for database index...")
            if progress_callback:
                progress_callback(frame_count, total_frames)
            
            # --- Model 1 Inference (Persons) ---
            results1 = model1.predict(frame, conf=0.15, verbose=False)
            for result in results1:
                boxes = result.boxes
                for idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                        
                    crop_filename = f"frame_{frame_count}_m1_person_{idx}.jpg"
                    crop_path = os.path.join(crops_dir, crop_filename)
                    cv2.imwrite(crop_path, crop)
                    
                    # Generate CLIP embedding
                    try:
                        pil_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        inputs = clip_processor(images=pil_image, return_tensors="pt").to(device)
                        with torch.no_grad():
                            # Extract raw pooled output from the returned BaseModelOutputWithPooling
                            image_features = clip_model.get_image_features(**inputs).pooler_output
                            # Normalize feature tensor to unit vector
                            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                            embedding = image_features.cpu().numpy().flatten()
                        embeddings_list.append(embedding)
                    except Exception as e:
                        print(f"Error extracting CLIP embedding for person object: {e}")
                        # Fallback zero embedding to align indices
                        embeddings_list.append(np.zeros(512, dtype=np.float32))
                    
                    saved_data.append({
                        "frame_id": frame_count,
                        "model": "best_v2",
                        "object_id": idx,
                        "class_id": cls,
                        "class_name": model1.names[cls],
                        "confidence": conf,
                        "crop_path": crop_path,
                        "bbox": [x1, y1, x2, y2],
                        "detected_color": "N/A"
                    })
                    
            # --- Model 2 Inference (Clothing) ---
            results2 = model2.predict(frame, conf=0.15, verbose=False)
            for result in results2:
                boxes = result.boxes
                for idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                        
                    # Color detection for clothing crop using OpenCV helper
                    color_name = get_dominant_color(crop)
                    
                    crop_filename = f"frame_{frame_count}_m2_cloth_{idx}.jpg"
                    crop_path = os.path.join(crops_dir, crop_filename)
                    cv2.imwrite(crop_path, crop)
                    
                    # Generate CLIP embedding
                    try:
                        pil_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        inputs = clip_processor(images=pil_image, return_tensors="pt").to(device)
                        with torch.no_grad():
                            # Extract raw pooled output from the returned BaseModelOutputWithPooling
                            image_features = clip_model.get_image_features(**inputs).pooler_output
                            # Normalize feature tensor to unit vector
                            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                            embedding = image_features.cpu().numpy().flatten()
                        embeddings_list.append(embedding)
                    except Exception as e:
                        print(f"Error extracting CLIP embedding for clothing object: {e}")
                        # Fallback zero embedding to align indices
                        embeddings_list.append(np.zeros(512, dtype=np.float32))
                    
                    saved_data.append({
                        "frame_id": frame_count,
                        "model": "cloth_YOLO",
                        "object_id": idx,
                        "class_id": cls,
                        "class_name": model2.names[cls],
                        "confidence": conf,
                        "crop_path": crop_path,
                        "bbox": [x1, y1, x2, y2],
                        "detected_color": color_name
                    })
                    
        frame_count += 1
        
    cap.release()
    print("Video processing complete.")
    
    # 5. Save metadata using Pandas
    df = pd.DataFrame(saved_data)
    csv_path = os.path.join(output_dir, "video_metadata.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved metadata to {csv_path}")
    
    # 6. Save embeddings to FAISS Vector Database
    if embeddings_list:
        embeddings_arr = np.array(embeddings_list, dtype=np.float32)
        # Normalize to unit vector for Cosine Similarity search
        faiss.normalize_L2(embeddings_arr)
        
        index = faiss.IndexFlatIP(512) # Dimension of CLIP features is 512
        index.add(embeddings_arr)
        
        index_path = os.path.join(output_dir, "embeddings.index")
        faiss.write_index(index, index_path)
        print(f"Saved {len(embeddings_list)} embeddings to FAISS index at {index_path}")

def play_video_with_yolo(video_path, yolo_model_path="best_v2.pt"):
    """
    Play a video and display YOLO predictions on it in real-time.
    """
    print(f"Loading YOLO model from: {yolo_model_path}")
    model = YOLO(yolo_model_path, task='detect')
    print(f"Model classes: {model.names}")
    
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLO prediction
        # Lowered conf to catch clothes, but removed class exclusion so it identifies both 'person' and clothing
        results = model.predict(frame, conf=0.5, verbose=False)
        
        # Plot the predictions on the frame
        annotated_frame = results[0].plot()
        
        # Display the frame
        cv2.imshow("YOLO Predictions", annotated_frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

def compare_models_side_by_side(video_path, model_path_1, model_path_2, output_video_path="output_combined.mp4", frame_skip=30):
    """
    Load two YOLO models, run prediction frame-by-frame (skipping frame_skip frames), display side-by-side in real-time,
    and save the combined comparison video to a file.
    """
    print(f"Loading Model 1 from: {model_path_1}")
    model1 = YOLO(model_path_1, task='detect')
    
    print(f"Loading Model 2 from: {model_path_2}")
    model2 = YOLO(model_path_2, task='detect')
    
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
        
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Info: Resolution {width}x{height}, FPS: {fps}, Total Frames: {total_frames}")
    
    # Determine resizing targets. We will resize each model's prediction window to 960x540
    # for a combined 1920x540 video. This maintains typical widescreen aspect ratios.
    target_w, target_h = 960, 540
    combined_w, combined_h = target_w * 2, target_h
    
    # Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (combined_w, combined_h))
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_skip == 0:
            # Inference for both models
            # Using a confidence threshold of 0.25 to avoid noise while catching detections
            results1 = model1.predict(frame, conf=0.25, verbose=False)
            results2 = model2.predict(frame, conf=0.25, verbose=False)
            
            # Plot annotated bounding boxes
            annotated_frame1 = results1[0].plot()
            annotated_frame2 = results2[0].plot()
            
            # Resize to fit side-by-side view
            resized_frame1 = cv2.resize(annotated_frame1, (target_w, target_h))
            resized_frame2 = cv2.resize(annotated_frame2, (target_w, target_h))
            
            # Draw labels / model name banners at the top of each sub-frame
            # Draw black rectangle banner
            cv2.rectangle(resized_frame1, (0, 0), (target_w, 40), (0, 0, 0), -1)
            cv2.putText(resized_frame1, "Model: best_v2 (Person & Gear)", (15, 28), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.rectangle(resized_frame2, (0, 0), (target_w, 40), (0, 0, 0), -1)
            cv2.putText(resized_frame2, "Model: cloth_YOLO (Clothing)", (15, 28), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Concatenate horizontally
            combined_frame = np.hstack((resized_frame1, resized_frame2))
            
            # Write to file
            out.write(combined_frame)
            
            # Show real-time window
            try:
                cv2.imshow("Dual Model Prediction Comparison", combined_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Video visualization interrupted by user.")
                    break
            except Exception as e:
                # Fallback for headless environments or terminal-only execution
                pass
            
        frame_idx += 1
        if frame_idx % 300 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames...")
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Combined side-by-side video successfully saved to: {output_video_path}")
    
if __name__ == "__main__":
    # Example usage:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(base_dir, "People in the park - 4K Slow motion stock video - 4K Stock Footage Pro (1080p, h264).mp4")
    
    # Paths to the YOLO models
    model_path_1 = os.path.join(base_dir, "best_v2.pt")
    model_path_2 = os.path.join(base_dir, "cloth_YOLO.pt")
    
    # 1. Run the pipeline to save cropped images and generate the metadata index CSV with detected colors
    print("--- Running Video Indexing Pipeline ---")
    process_video_pipeline(
        video_path=video_path,
        yolo_model_path_1=model_path_1,
        yolo_model_path_2=model_path_2,
        output_dir=os.path.join(base_dir, "output"),
        frame_skip=30
    )
    
    # 2. To play and compare the video with both YOLO predictions side-by-side:
    # print("\n--- Running Real-time Comparison Visualization ---")
    # output_video_path = "/Users/riishabhjain/Desktop/AI_VIDEO_SEARCH/output_combined.mp4"
    # compare_models_side_by_side(
    #     video_path=video_path,
    #     model_path_1=model_path_1,
    #     model_path_2=model_path_2,
    #     output_video_path=output_video_path,
    #     frame_skip=30
    # )
