import streamlit as st
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
import tempfile
import os
import time
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict
from streamlit.errors import StreamlitAPIException

# --- Configuration ---
ROBOFLOW_API_KEY = "8AwyUdaO3pdVgnH07lVn" # WARNING: Hardcoded API Key
PITCH_MODEL_ID = "football-field-detection-f07vi-fmhti/1"
API_URL = "https://serverless.roboflow.com"
TRACKER_MODEL_PATH = 'modelo_deteccion_futbol.pt'

# --- Constants ---
REFERENCE_KEYPOINT_MAP = {
    '15': 'top_penalty_spot', '19': 'center_spot', '17': 'bottom_penalty_spot',
    '14': 'top_left_corner', '20': 'top_right_corner', '30': 'bottom_left_corner',
    '23': 'bottom_right_corner' # !!! VERIFY THESE !!!
}
PITCH_CLASS_NAME = 'pitch' # The class name your Roboflow model uses for the pitch itself

# --- Drawing Configuration ---
PITCH_BOX_COLOR=(0, 255, 0); PITCH_KEYPOINT_COLOR=(255, 0, 0);
TRACKING_COLORS=defaultdict(lambda: (np.random.randint(50, 200), np.random.randint(50, 200), np.random.randint(50, 200)))
TEXT_COLOR=(255, 255, 255); BOX_THICKNESS=2; KEYPOINT_RADIUS=5
FONT_SCALE=0.6; FONT=cv2.FONT_HERSHEY_SIMPLEX

# --- Streamlit Page Config ---
st.set_page_config(layout="wide")

# --- Model Loading ---
@st.cache_resource
def load_models(api_url, api_key, tracker_path):
    """Loads Roboflow client and YOLO tracker model."""
    pitch_client, tracker_model = None, None
    try:
        pitch_client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
    except Exception as e: st.error(f"Roboflow client error: {e}")
    try:
        if os.path.exists(tracker_path):
            tracker_model = YOLO(tracker_path)
        else:
             st.error(f"Tracker model file not found: {tracker_path}")
    except Exception as e: st.error(f"Tracker model loading error: {e}")
    return pitch_client, tracker_model

CLIENT, TRACKER_MODEL = load_models(API_URL, ROBOFLOW_API_KEY, TRACKER_MODEL_PATH)

# Display Status & Check Models
if CLIENT: st.sidebar.success("Roboflow client OK.")
if TRACKER_MODEL: st.sidebar.success(f"Tracker model OK.")
else: st.sidebar.error("Tracker model FAILED to load.")
if CLIENT is None or TRACKER_MODEL is None:
    st.error("One or more models failed to load. Cannot continue.")
    st.stop()
TRACKER_CLASS_NAMES = TRACKER_MODEL.names

# --- Helper Functions ---
def get_real_world_ref_points(pitch_length_m, pitch_width_m):
    """Defines real-world coordinates for reference points."""
    ref_points_m = {
        'bottom_left_corner': (0, 0), 'bottom_right_corner': (pitch_length_m, 0),
        'top_left_corner': (0, pitch_width_m), 'top_right_corner': (pitch_length_m, pitch_width_m),
        'center_spot': (pitch_length_m / 2, pitch_width_m / 2),
        'bottom_penalty_spot': (11, pitch_width_m / 2),
        'top_penalty_spot': (pitch_length_m - 11, pitch_width_m / 2), }
    return {k: v for k, v in ref_points_m.items() if k in REFERENCE_KEYPOINT_MAP.values()}

def calculate_homography(pitch_pred, ref_points, img_shape, keypoint_conf_threshold):
    """Calculates homography matrix H. Returns H if successful, None otherwise."""
    if not pitch_pred or 'keypoints' not in pitch_pred or not pitch_pred['keypoints']: return None
    img_h, img_w = img_shape[:2]; px_pts, wd_pts, det_kps = [], [], {}
    for kp in pitch_pred['keypoints']:
        kp_cls, kp_conf = kp.get('class'), kp.get('confidence', 0)
        if kp_cls in REFERENCE_KEYPOINT_MAP and kp_conf >= keypoint_conf_threshold:
            det_kps[REFERENCE_KEYPOINT_MAP[kp_cls]] = (int(kp['x']), int(kp['y']))
    matched_points_count = 0
    for name, wd_coord in ref_points.items():
        if name in det_kps:
            px, py = det_kps[name]
            if 0 <= px < img_w and 0 <= py < img_h:
                px_pts.append((px, py)); wd_pts.append(wd_coord); matched_points_count += 1
    if matched_points_count >= 4:
        try:
            H, _ = cv2.findHomography(np.array(px_pts, np.float32), np.array(wd_pts, np.float32), cv2.RANSAC, 5.0)
            if H is not None and 1e-6 < abs(np.linalg.det(H)) < 1e6 : return H
        except cv2.error: pass
    return None

def transform_coords(px_coords, H):
    """Transforms pixel coordinates to world coordinates using Homography H."""
    if H is None: return None
    try:
        res = cv2.perspectiveTransform(np.array([[px_coords]], dtype=np.float32), H)
        if res is not None: return res[0][0]
    except Exception: pass
    return None

# --- Drawing Functions ---
def draw_pitch_box_and_label(frame, pitch_pred):
    """Draws only the bounding box and label for the pitch."""
    if not pitch_pred: return frame
    h, w, _ = frame.shape
    xc, yc = int(pitch_pred['x']), int(pitch_pred['y'])
    bw, bh = int(pitch_pred['width']), int(pitch_pred['height'])
    conf, cls_nm = pitch_pred.get('confidence'), pitch_pred.get('class', PITCH_CLASS_NAME)
    x1, y1 = max(0, xc - bw // 2), max(0, yc - bh // 2)
    x2, y2 = min(w - 1, xc + bw // 2), min(h - 1, yc + bh // 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), PITCH_BOX_COLOR, BOX_THICKNESS)
    lbl = f"{cls_nm}: {conf:.2f}" if conf else cls_nm
    (tw, th), _ = cv2.getTextSize(lbl, FONT, FONT_SCALE, BOX_THICKNESS)
    cv2.rectangle(frame, (x1, y1 - th - 5), (x1 + tw, y1), PITCH_BOX_COLOR, -1)
    cv2.putText(frame, lbl, (x1, y1 - 5), FONT, FONT_SCALE, TEXT_COLOR, cv2.LINE_AA)
    return frame

def draw_pitch_keypoints(frame, pitch_pred, kp_conf_thresh_display):
    """Draws keypoints if they exist and meet the display confidence."""
    if not pitch_pred or 'keypoints' not in pitch_pred or not pitch_pred['keypoints']:
        return frame
    h, w, _ = frame.shape
    for pt in pitch_pred['keypoints']:
        kp_conf = pt.get('confidence', 0)
        if kp_conf >= kp_conf_thresh_display:
            kpx, kpy = max(0, min(w-1, int(pt['x']))), max(0, min(h-1, int(pt['y'])))
            cv2.circle(frame, (kpx, kpy), KEYPOINT_RADIUS, PITCH_KEYPOINT_COLOR, -1)
    return frame

def draw_tracked_objects(frame, track_res, H):
    """Draws bounding boxes and labels for tracked objects."""
    if not track_res or not track_res[0].boxes: return frame
    boxes = track_res[0].boxes.xyxy.cpu().numpy().astype(int)
    ids = track_res[0].boxes.id.cpu().numpy().astype(int) if track_res[0].boxes.id is not None else None
    confs = track_res[0].boxes.conf.cpu().numpy(); classes = track_res[0].boxes.cls.cpu().numpy().astype(int)
    if ids is None: return frame

    for box, tid, conf, cid in zip(boxes, ids, confs, classes):
        x1, y1, x2, y2 = box; color = TRACKING_COLORS[tid]
        cls_nm = TRACKER_CLASS_NAMES.get(cid, f"Cls:{cid}")
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        px_x, px_y = (x1 + x2) // 2, y2 # Bottom center
        wd_coords = transform_coords((px_x, px_y), H) # Attempt transform
        lbl = f"ID:{tid} {cls_nm}"
        if wd_coords is not None: lbl += f" ({wd_coords[0]:.1f},{wd_coords[1]:.1f}m)"
        (tw, th), _ = cv2.getTextSize(lbl, FONT, FONT_SCALE, BOX_THICKNESS)
        cv2.rectangle(frame, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
        cv2.putText(frame, lbl, (x1, y1 - 5), FONT, FONT_SCALE, TEXT_COLOR, cv2.LINE_AA)
    return frame

# --- Core Processing Functions ---
def run_pitch_detection(client, frame):
    """Runs Roboflow inference, handles temp file and errors."""
    pitch_result_raw = None
    temp_frame_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tf:
            cv2.imwrite(tf.name, frame)
            temp_frame_path = tf.name
        # *** REMOVED confidence argument from infer call ***
        pitch_result_raw = client.infer(temp_frame_path, model_id=PITCH_MODEL_ID)
    except Exception as e:
        st.warning(f"Roboflow API Error: {e}", icon="🤖") # Use icon for clarity
    finally:
        if temp_frame_path and os.path.exists(temp_frame_path):
            os.remove(temp_frame_path)
    return pitch_result_raw

def extract_main_pitch_prediction(pitch_result_raw, confidence_threshold):
    """Extracts the highest confidence 'pitch' prediction above threshold."""
    if not pitch_result_raw or 'predictions' not in pitch_result_raw:
        return None
    
    best_pitch_pred = None
    max_conf = -1.0

    for pred in pitch_result_raw['predictions']:
        if pred.get('class') == PITCH_CLASS_NAME and pred.get('confidence', 0) >= confidence_threshold:
             if pred['confidence'] > max_conf:
                  max_conf = pred['confidence']
                  best_pitch_pred = pred
                  
    return best_pitch_pred

def update_homography_matrix(pitch_pred_data, ref_points, img_shape, homography_conf_thresh):
    """Calculates and returns a new homography matrix if possible."""
    return calculate_homography(pitch_pred_data, ref_points, img_shape, homography_conf_thresh)

def run_tracking(tracker_model, frame, tracker_conf_thresh):
    """Runs YOLO tracking on the frame."""
    try:
        return tracker_model.track(frame, persist=True, verbose=False, conf=tracker_conf_thresh)
    except Exception as e:
        st.warning(f"Tracker Error: {e}", icon="🎯")
        return None

def append_tracking_data(data_list, track_results, H, frame_idx):
    """Appends data for tracked objects to the list."""
    if not track_results or not track_results[0].boxes: return
    boxes = track_results[0].boxes.xyxy.cpu().numpy().astype(int)
    ids = track_results[0].boxes.id.cpu().numpy().astype(int) if track_results[0].boxes.id is not None else None
    confs = track_results[0].boxes.conf.cpu().numpy(); classes = track_results[0].boxes.cls.cpu().numpy().astype(int)
    if ids is None: return

    for box, tid, conf, cid in zip(boxes, ids, confs, classes):
        x1, y1, x2, y2 = box
        cls_nm = TRACKER_CLASS_NAMES.get(cid, f"Cls:{cid}")
        px_x, px_y = (x1 + x2) // 2, y2
        wd_coords = transform_coords((px_x, px_y), H)
        wx, wy = (wd_coords[0], wd_coords[1]) if wd_coords is not None else (None, None)
        data_list.append({
            'frame': frame_idx, 'track_id': tid, 'class_id': cid, 'class_name': cls_nm,
            'confidence': conf, 'bbox_x1': x1, 'bbox_y1': y1, 'bbox_x2': x2, 'bbox_y2': y2,
            'pixel_x': px_x, 'pixel_y': px_y, 'world_x_m': wx, 'world_y_m': wy})

def process_frame_visuals(frame, pitch_data, track_data, H, kp_display_conf):
    """Draws all visualizations onto the frame."""
    processed_frame = frame.copy()
    processed_frame = draw_pitch_box_and_label(processed_frame, pitch_data)
    processed_frame = draw_pitch_keypoints(processed_frame, pitch_data, kp_display_conf)
    processed_frame = draw_tracked_objects(processed_frame, track_data, H)
    return processed_frame

# --- Streamlit UI Setup ---
st.title("⚽ Football Pitch & Player/Ball Tracker")
st.write("Upload video, set dimensions, track objects.")

# Sidebar...
st.sidebar.markdown("---"); st.sidebar.header("Pitch Dimensions (m)")
pitch_length = st.sidebar.number_input("Pitch Length", 50.0, 150.0, 105.0, 1.0)
pitch_width = st.sidebar.number_input("Pitch Width", 30.0, 100.0, 68.0, 1.0)
st.sidebar.markdown("---"); st.sidebar.header("Confidence Thresholds")
# NEW: Confidence for accepting the main pitch detection result
pitch_detect_conf = st.sidebar.slider("Pitch Detection Confidence", 0.0, 1.0, 0.5, 0.05)
pitch_kp_conf_display = st.sidebar.slider("Pitch Keypoint Display Conf", 0.0, 1.0, 0.3, 0.05)
pitch_kp_conf_homography = st.sidebar.slider("Pitch Keypoint Homography Conf", 0.0, 1.0, 0.2, 0.05) # Default 0.2?
tracker_conf_threshold = st.sidebar.slider("Tracker Conf", 0.0, 1.0, 0.4, 0.05)
st.sidebar.markdown("---"); st.sidebar.info(f"Pitch Model: `{PITCH_MODEL_ID}`")
st.sidebar.info(f"Tracker Model: `{os.path.basename(TRACKER_MODEL_PATH)}`")
st.sidebar.warning("⚠️ API Key/Model Paths might be hardcoded.")

# --- Session State Initialization ---
default_state = {
    'stop_processing': False, 'processing_started': False,
    'processed_video_path': None, 'collected_data_df': None,
    'last_processed_frame_rgb': None, 'upload_key': 0,
    'current_status_msg': "", 'show_results': False,
    'homography_active': False, 'homography_matrix': None, # Store H in state
    'start_time': 0.0, 'processed_frames': 0, 'collected_data': [] # Store processing state
}
for key, value in default_state.items():
    if key not in st.session_state: st.session_state[key] = value

# --- Main Area Placeholders ---
output_video_placeholder = st.empty()
control_placeholder = st.empty()
data_table_placeholder = st.empty()
download_placeholder = st.container()

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Choose a video file", type=["mp4", "avi", "mov", "mkv"],
    key=f"uploader_{st.session_state.upload_key}" )

# --- Button Definition & Status Display ---
homography_status_text = "H Status: " + ("✅ On" if st.session_state.homography_active else "❌ Off")
if st.session_state.processing_started:
    with control_placeholder.container():
        col1, col2 = st.columns([3,1])
        with col1: st.info(f"{st.session_state.current_status_msg} | {homography_status_text}")
        with col2:
            if st.button("⏹️ Stop", key="stop_button"):
                st.session_state.stop_processing = True
                st.session_state.current_status_msg = "Stop requested..."

# --- Processing Logic ---
if uploaded_file is not None and not st.session_state.processing_started and not st.session_state.show_results:
    # Start Processing - Reset State
    st.session_state.processing_started = True
    st.session_state.stop_processing = False
    st.session_state.processed_video_path = None
    st.session_state.collected_data_df = None
    st.session_state.last_processed_frame_rgb = None
    st.session_state.current_status_msg = "Initializing..."
    st.session_state.show_results = False
    st.session_state.homography_active = False
    st.session_state.homography_matrix = None # Reset H
    st.session_state.processed_frames = 0 # Reset frame count
    st.session_state.collected_data = [] # Reset data list
    st.session_state.start_time = time.time() # Reset timer
    output_video_placeholder.empty()
    data_table_placeholder.empty()
    download_placeholder.empty()
    st.session_state.uploaded_file_bytes = uploaded_file.getvalue()
    st.rerun() # Show stop button and initial status

# --- Actual processing happens here ---
if st.session_state.processing_started:
    temp_input_video_path = None; cap = None; video_writer = None
    try:
        # Initialization using state
        if 'uploaded_file_bytes' in st.session_state and st.session_state.uploaded_file_bytes:
             tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
             tfile.write(st.session_state.uploaded_file_bytes)
             temp_input_video_path = tfile.name
        else: raise ValueError("File data error")

        real_world_ref_points = get_real_world_ref_points(pitch_length, pitch_width)
        if not real_world_ref_points: raise ValueError("Reference points error")
        cap = cv2.VideoCapture(temp_input_video_path)
        if not cap.isOpened(): raise ValueError("Video open error")
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if st.session_state.processed_video_path is None:
             st.session_state.processed_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='_output.mp4').name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v'); video_writer = cv2.VideoWriter(st.session_state.processed_video_path, fourcc, fps, (frame_width, frame_height))
        if not video_writer.isOpened(): raise ValueError("Video writer error")

        # Retrieve processing state
        processed_frames = st.session_state.processed_frames
        collected_data = st.session_state.collected_data # Use state list directly
        homography_matrix = st.session_state.homography_matrix # Use state H

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        progress_bar = st.progress(processed_frames / total_frames)
        st.session_state.current_status_msg = "Processing..." # Update status

        # --- Processing Loop ---
        while cap.isOpened():
            if st.session_state.stop_processing: break
            ret, frame = cap.read();
            if not ret: break

            current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # 1. Pitch Detection
            pitch_result_raw = run_pitch_detection(CLIENT, frame)
            pitch_data = extract_main_pitch_prediction(pitch_result_raw, pitch_detect_conf) # Use pitch detect confidence

            # 2. Homography Update
            frame_homography_active = False
            if pitch_data:
                new_H = update_homography_matrix(pitch_data, real_world_ref_points, frame.shape, pitch_kp_conf_homography)
                if new_H is not None:
                    homography_matrix = new_H # Update local H for this frame
                    st.session_state.homography_matrix = homography_matrix # Store in state
                    frame_homography_active = True

            # Update homography active status if changed (no immediate rerun)
            if frame_homography_active != st.session_state.homography_active:
                 st.session_state.homography_active = frame_homography_active

            # 3. Tracking
            track_results = run_tracking(TRACKER_MODEL, frame, tracker_conf_threshold)

            # 4. Append Data (uses current homography_matrix)
            append_tracking_data(collected_data, track_results, homography_matrix, current_frame_num)

            # 5. Process Visuals (uses current H and pitch_data)
            processed_frame_visuals = process_frame_visuals(frame, pitch_data, track_results, homography_matrix, pitch_kp_conf_display)

            # 6. Display & Write
            frame_rgb = cv2.cvtColor(processed_frame_visuals, cv2.COLOR_BGR2RGB)
            st.session_state.last_processed_frame_rgb = frame_rgb
            output_video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            if video_writer.isOpened(): video_writer.write(processed_frame_visuals)

            # 7. Update Progress State
            processed_frames += 1; st.session_state.processed_frames = processed_frames
            progress = min(1.0, processed_frames / total_frames)
            try: progress_bar.progress(progress)
            except StreamlitAPIException: pass # Ignore if bar removed

            # 8. Update data table preview
            if current_frame_num % 30 == 0 and collected_data:
                with data_table_placeholder: st.dataframe(pd.DataFrame(collected_data).tail(20))

        # --- End of Loop ---
    except Exception as e:
        st.error(f"Error during processing loop: {e}")
        st.session_state.current_status_msg = f"Error: {e}"
        st.session_state.stop_processing = True # Treat errors as stop
    finally:
        # --- Cleanup and Final State Update ---
        if cap is not None and cap.isOpened(): cap.release()
        if video_writer is not None and video_writer.isOpened(): video_writer.release()
        if temp_input_video_path and os.path.exists(temp_input_video_path):
             try: os.remove(temp_input_video_path)
             except Exception: pass
        if 'uploaded_file_bytes' in st.session_state: del st.session_state['uploaded_file_bytes']

        # Process final data from state list
        final_collected_data = st.session_state.get('collected_data', [])
        if final_collected_data: st.session_state.collected_data_df = pd.DataFrame(final_collected_data)
        else: st.session_state.collected_data_df = None

        # Set final status...
        end_time = time.time(); start_time = st.session_state.get('start_time', 0.0)
        processing_time = end_time - start_time; final_frames = st.session_state.get('processed_frames', 0)
        current_msg = st.session_state.current_status_msg
        if not current_msg.startswith("Error") and not current_msg.startswith("Stop"):
             status_prefix = "Processing stopped" if st.session_state.stop_processing else "Processing finished"
             st.session_state.current_status_msg = f"{status_prefix} after {processing_time:.2f}s ({final_frames} frames)."

        # Prepare for results display...
        st.session_state.processing_started = False; st.session_state.show_results = True
        st.session_state.upload_key += 1
        # Don't clear processing state here, keep it for results display

        # Clear processing UI elements
        if 'progress_bar' in locals() and progress_bar is not None: progress_bar.empty()
        control_placeholder.empty()

        # Rerun to display results section
        st.rerun()


# --- Results Display and Download Section ---
if not st.session_state.processing_started and st.session_state.show_results:
    # Display final status...
    final_msg = st.session_state.current_status_msg
    h_status = "H Status: " + ("✅ Active" if st.session_state.homography_active else "❌ Inactive") # Use final H status
    full_final_msg = f"{final_msg} | {h_status}" if final_msg else h_status
    if final_msg.startswith("Error"): st.error(full_final_msg)
    else: st.success(full_final_msg)

    # Display last frame...
    if st.session_state.last_processed_frame_rgb is not None:
        output_video_placeholder.image( st.session_state.last_processed_frame_rgb, caption="Last Processed Frame",
            channels="RGB", use_container_width=True )
    else: output_video_placeholder.info("No frame processed/stored.")

    # Display data table...
    df = st.session_state.collected_data_df
    if df is not None and not df.empty:
         with data_table_placeholder:
             st.header("Collected Tracking Data (Final)")
             st.dataframe(df)
             valid_world_coords = df['world_x_m'].notna().sum(); total_detections = len(df)
             if total_detections > 0:
                 st.caption(f"World coords found for {valid_world_coords}/{total_detections} detections ({valid_world_coords/total_detections:.1%}).")
             else: st.caption("No detections recorded.")
    else: data_table_placeholder.info("No tracking data collected.")

    # Display download buttons...
    with download_placeholder:
        st.subheader("Download Results")
        dl_appeared = False
        # CSV
        if df is not None and not df.empty:
            dl_appeared = True; csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(label="⬇️ Data (.csv)", data=csv_data,
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_data.csv" if uploaded_file else "data.csv",
                mime="text/csv", key="dl-csv" )
        # Video
        vid_path = st.session_state.processed_video_path
        vid_exists = vid_path and os.path.exists(vid_path) and os.path.getsize(vid_path) > 1024
        if vid_exists:
            try:
                with open(vid_path, "rb") as f: video_bytes = f.read()
                dl_appeared = True
                st.download_button(label="⬇️ Video (.mp4)", data=video_bytes,
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_processed.mp4" if uploaded_file else "video.mp4",
                    mime="video/mp4", key="dl-vid" )
            except Exception as e: st.error(f"Error reading video for download: {e}")
        elif st.session_state.processed_video_path: st.warning("Processed video missing or empty.")
        if not dl_appeared: st.info("No results to download.")

    # Clear Results Button...
    if st.button("Clear Results & Upload New", key="clear_results"):
        # Reset relevant state variables
        st.session_state.show_results = False
        st.session_state.last_processed_frame_rgb = None
        st.session_state.collected_data_df = None
        st.session_state.current_status_msg = ""
        st.session_state.homography_active = False
        st.session_state.homography_matrix = None
        st.session_state.processed_frames = 0
        st.session_state.collected_data = []
        vid_path = st.session_state.processed_video_path
        if vid_path and os.path.exists(vid_path):
            try: os.remove(vid_path)
            except Exception: pass
        st.session_state.processed_video_path = None
        if 'uploaded_file_bytes' in st.session_state: del st.session_state['uploaded_file_bytes']
        st.rerun()