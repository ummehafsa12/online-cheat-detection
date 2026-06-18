
import cv2
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine
import threading

# ==========================================
# REAL-TIME FACE VERIFICATION SYSTEM
# ==========================================
# 1. MODEL SELECTION: FaceNet
# Why FaceNet? FaceNet utilizes a robust deep convolutional neural network 
# trained to map faces into a highly distinct 128-dimensional Euclidean space. 
# It natively achieves ~99.63% accuracy on the LFW dataset. Distances between 
# these embeddings directly correspond to face similarity, making it perfect 
# and extremely reliable for direct verification tasks without heavy overhead.
# ========================================

MODEL_NAME = "Facenet"
REFERENCE_IMAGE_PATH = "profile.jpg" # ⚠️ CHANGE THIS to your reference image path!
DISTANCE_METRIC = "cosine"
THRESHOLD_DISTANCE = 0.40 # Standard strict FaceNet cosine threshold
PROCESS_EVERY_N_FRAMES = 10 # Process every N frames to maintain smooth 30+ FPS webcam UI

# Global variables for cross-thread data sharing
verification_result = "Analyzing..."
similarity_score = 0.0
is_processing = False
face_box = None

def get_embedding(img_path_or_array):
    """
    Generate FaceNet embedding for an image or frame.
    Returns: (embedding_vector, facial_area_dict)
    """
    try:
        # enforce_detection=True ensures we error out elegantly if no face is found
        objs = DeepFace.represent(
            img_path=img_path_or_array, 
            model_name=MODEL_NAME, 
            enforce_detection=True
        )
        if objs and len(objs) > 0:
            return objs[0]["embedding"], objs[0]["facial_area"]
    except ValueError:
        pass # DeepFace throws ValueError if no face is detected
    except Exception as e:
        print(f"Embedding error: {e}")
    return None, None

def verify_face_thread(frame, ref_embedding):
    """
    Background thread to process the current frame.
    This prevents the heavy FaceNet model from freezing the webcam feed.
    """
    global verification_result, similarity_score, is_processing, face_box
    
    # 1. Get embedding for the current webcam frame
    frame_embedding, area = get_embedding(frame)
    
    if frame_embedding is None:
        verification_result = "No face detected"
        similarity_score = 0.0
        face_box = None
    else:
        # 2. Compare embeddings using Cosine distance
        distance = cosine(ref_embedding, frame_embedding)
        
        # 3. Convert distance to a 0.0 - 1.0 similarity score.
        # FaceNet Cosine distance ranges from 0 (perfect match) to 1.0+.
        # A standard threshold for FaceNet is 0.40.
        sim = max(0.0, 1.0 - distance)
            
        similarity_score = sim
        face_box = area
        
        # 4. Decision Logic (Strict > 99% accuracy threshold)
        if distance <= THRESHOLD_DISTANCE:
            verification_result = "Face Verified"
        else:
            verification_result = "Not Verified"
            
    # Mark thread as finished
    is_processing = False

def main():
    global is_processing
    
    print(f"[*] Loading reference image: {REFERENCE_IMAGE_PATH}")
    print(f"[*] Initializing {MODEL_NAME} model. This might take a moment...")
    
    # Load profile image (reference image)
    ref_embedding, _ = get_embedding(REFERENCE_IMAGE_PATH)
    if ref_embedding is None:
        print(f"\n[!] ERROR: Could not detect a face in '{REFERENCE_IMAGE_PATH}'.")
        print("[!] Please ensure the image exists and contains a clear face.")
        return
        
    print("[*] Reference face embedding loaded successfully!")
    
    # Open webcam using OpenCV
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] ERROR: Cannot open webcam")
        return
        
    frame_count = 0
    print("[*] Starting live webcam feed... Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Failed to grab frame")
            break
            
        # To optimize for real-time, process every Nth frame, and only if previous thread finished
        if frame_count % PROCESS_EVERY_N_FRAMES == 0 and not is_processing:
            is_processing = True
            # Pass a copy of the frame to the thread to avoid memory collision
            threading.Thread(
                target=verify_face_thread, 
                args=(frame.copy(), ref_embedding), 
                daemon=True
            ).start()
            
        # Visual bounding box mapping
        if face_box is not None:
            x, y = face_box['x'], face_box['y']
            w, h = face_box['w'], face_box['h']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
            
        # Determine UI color based on result
        color = (0, 0, 255) # Red for Not Verified
        if verification_result == "Face Verified":
            color = (0, 255, 0) # Green
        elif verification_result == "No face detected":
            color = (0, 255, 255) # Yellow
            
        # Display Output Verification Result
        cv2.putText(frame, verification_result, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        
        # Display Similarity Score
        if verification_result != "No face detected" and verification_result != "Analyzing...":
            cv2.putText(frame, f"Sim Score: {similarity_score:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        # Instructions
        cv2.putText(frame, "Press 'q' to quit", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Show webcam feed
        cv2.imshow("Real-Time Face Verification (FaceNet)", frame)
        
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()