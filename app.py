import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import math

# --- Configuración de la Página (PRIMER COMANDO STREAMLIT) ---
st.set_page_config(page_title="Detector Fútbol Completo", page_icon="⚽", layout="wide")
# --------------------------------------------------------------

# --- Helper Functions ---
def hex_to_rgb(hex_color):
    """Convierte un color HEX (ej: '#FF0000') a una tupla RGB (0-255)."""
    hex_color = hex_color.lstrip('#'); hlen = len(hex_color)
    try:
        if hlen == 3: return tuple(int(hex_color[i]*2, 16) for i in range(hlen))
        elif hlen == 6: return tuple(int(hex_color[i:i+2], 16) for i in range(0, hlen, 2))
        else: raise ValueError("Longitud HEX inválida")
    except ValueError: st.error(f"Error convirtiendo HEX: '{hex_color}'"); return (0, 0, 0)

def rgb_to_bgr(rgb_color):
    """Convierte una tupla RGB a BGR."""
    return (rgb_color[2], rgb_color[1], rgb_color[0])

def color_distance(color1_rgb, color2_rgb):
    """Calcula la distancia Euclidiana entre dos colores RGB."""
    if color1_rgb is None or color2_rgb is None: return float('inf')
    c1 = np.array(color1_rgb, dtype=float); c2 = np.array(color2_rgb, dtype=float)
    return np.linalg.norm(c1 - c2)

def calculate_center(box):
    """Calcula el centro (x, y) de una caja [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = box; return (x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2)

def initialize_kalman():
    """Inicializa y devuelve un objeto KalmanFilter configurado para rastreo 2D."""
    kf = cv2.KalmanFilter(4, 2) # Estado: [x, y, vx, vy], Medida: [x, y]
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    dt = 1; kf.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    # Valores iniciales de ruido (se pueden sobrescribir desde la UI)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5e-1
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 1000 # Incertidumbre inicial alta
    kf.statePost = np.zeros((4, 1), dtype=np.float32)
    return kf

def get_dominant_color(image, k=3, n_init=10, max_pixels=1000,
                       exclude_color1_rgb=None,
                       exclude_color2_rgb=None,
                       exclude_tolerance=50,
                       min_pixels_after_exclude=20):
    """
    Encuentra el color dominante en una imagen usando K-Means,
    opcionalmente excluyendo píxeles cercanos a dos colores especificados.
    """
    try:
        if image is None or image.shape[0] < 2 or image.shape[1] < 2: return None
        h, w, _ = image.shape; pixels_total = h * w
        if pixels_total == 0: return None

        image_resized = image.copy()
        if pixels_total > max_pixels:
            scale = np.sqrt(max_pixels / pixels_total)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        image_rgb_float = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(np.float32)
        if len(image_rgb_float) == 0: return None

        pixels_to_process = image_rgb_float
        num_pixels_initial = len(pixels_to_process)

        # Filtrar colores excluidos
        if exclude_color1_rgb is not None or exclude_color2_rgb is not None:
            keep_mask = np.ones(num_pixels_initial, dtype=bool)

            if exclude_color1_rgb is not None:
                color1_np = np.array(exclude_color1_rgb, dtype=np.float32)
                distances_sq1 = np.sum((pixels_to_process - color1_np)**2, axis=1)
                keep_mask &= (distances_sq1 > exclude_tolerance**2)

            if exclude_color2_rgb is not None:
                color2_np = np.array(exclude_color2_rgb, dtype=np.float32)
                distances_sq2 = np.sum((pixels_to_process - color2_np)**2, axis=1)
                keep_mask &= (distances_sq2 > exclude_tolerance**2)

            pixels_to_process = pixels_to_process[keep_mask]

        if len(pixels_to_process) < min_pixels_after_exclude: return None

        if len(pixels_to_process) > max_pixels:
             n_samples_km = min(max_pixels, len(pixels_to_process))
             pixels_to_process = shuffle(pixels_to_process, random_state=0, n_samples=n_samples_km)

        actual_k = min(k, len(pixels_to_process))
        if actual_k < 1: return None
        kmeans_n_init = n_init if actual_k > 1 else 1
        kmeans = KMeans(n_clusters=actual_k, n_init=kmeans_n_init, random_state=0, algorithm='lloyd')
        kmeans.fit(pixels_to_process)

        unique, counts = np.unique(kmeans.labels_, return_counts=True)
        dominant_cluster_index = unique[np.argmax(counts)]
        dominant_color_rgb = kmeans.cluster_centers_[dominant_cluster_index].astype(int)

        return tuple(dominant_color_rgb)
    except Exception as e: return None
# --- Fin Helper Functions ---

# --- Carga del Modelo (DESPUÉS de set_page_config) ---
@st.cache_resource
def load_model(model_path):
    """Carga el modelo YOLO desde la ruta especificada."""
    try:
        if not os.path.exists(model_path):
             st.error(f"Error: No se encontró el archivo del modelo en: '{model_path}'")
             st.error("Asegúrate de que 'modelo_deteccion_futbol.pt' está en el mismo dir o usa ruta completa.")
             st.stop()
        model = YOLO(model_path)
        st.success(f"Modelo '{os.path.basename(model_path)}' cargado.")
        if hasattr(model, 'names') and isinstance(model.names, dict):
             st.write("Clases detectadas:", model.names)
             return model
        else: st.error("Modelo sin atributo 'names'."); st.stop()
    except Exception as e: st.error(f"Error cargando modelo: {e}"); st.stop()

MODEL_PATH = 'modelo_deteccion_futbol.pt'
model = load_model(MODEL_PATH)
# -----------------------------------------------------

# --- Constantes y Definiciones de Clases ---
PLAYER_CLASS_NAME = 'player'
GOALKEEPER_CLASS_NAME = 'goalkeeper'
BALL_CLASS_NAME = 'ball'
REFEREE_CLASS_NAME = 'referee'
CLASS_NAMES_TO_IDS = {name.lower(): k for k, name in model.names.items()}
PLAYER_CLS_ID = CLASS_NAMES_TO_IDS.get(PLAYER_CLASS_NAME.lower())
GOALKEEPER_CLS_ID = CLASS_NAMES_TO_IDS.get(GOALKEEPER_CLASS_NAME.lower())
BALL_CLS_ID = CLASS_NAMES_TO_IDS.get(BALL_CLASS_NAME.lower())
if BALL_CLS_ID is None: st.warning(f"Clase '{BALL_CLASS_NAME}' no encontrada en el modelo.")

DEFAULT_BOX_COLOR_BGR = (0, 255, 0); UNKNOWN_PLAYER_COLOR_BGR = (128, 128, 128)
DETECTED_BALL_COLOR_BGR = (0, 255, 0); PREDICTED_BALL_COLOR_BGR = (0, 165, 255) # Verde y Naranja
PREDICTED_BALL_BOX_SIZE_HALF = 8
# ------------------------------------------

# --- Título y Descripción ---
st.title("⚽ Detector Fútbol: Equipos, Posesión %, Pases por Equipo")
st.write("Excluye colores de césped para detectar equipos, calcula posesión y cuenta pases.")
# ----------------------------------------------------

# --- Configuración Sidebar ---
st.sidebar.header("Configuración")

st.sidebar.subheader("Equipos")
team_1_name = st.sidebar.text_input("Nombre Equipo 1", "Equipo Local")
team_1_hex = st.sidebar.color_picker(f'Color {team_1_name}', "#9DE380")
team_1_color_rgb = hex_to_rgb(team_1_hex); TEAM_1_BOX_COLOR_BGR = rgb_to_bgr(team_1_color_rgb)
st.sidebar.divider()
team_2_name = st.sidebar.text_input("Nombre Equipo 2", "Equipo Visitante")
team_2_hex = st.sidebar.color_picker(f'Color {team_2_name}', "#E7F1FE")
team_2_color_rgb = hex_to_rgb(team_2_hex); TEAM_2_BOX_COLOR_BGR = rgb_to_bgr(team_2_color_rgb)
st.sidebar.info(f"{team_1_name} RGB: {team_1_color_rgb}")
st.sidebar.info(f"{team_2_name} RGB: {team_2_color_rgb}")

st.sidebar.subheader("Parámetros Detección")
confidence_threshold = st.sidebar.slider("Confianza Mínima", 0.0, 1.0, 0.4, 0.05)
color_distance_threshold = st.sidebar.slider("Tolerancia Color Equipo", 0, 255, 130, 5)
kmeans_k = st.sidebar.slider("Clusters K (Color)", 1, 8, 3, 1)

st.sidebar.subheader("Exclusión de Colores (Césped)")
enable_grass_exclusion = st.sidebar.checkbox("Habilitar exclusión césped", value=True)
grass_color1_hex = st.sidebar.color_picker("Color Césped 1 (Ej: Sol)", "#559944")
grass_color2_hex = st.sidebar.color_picker("Color Césped 2 (Ej: Sombra)", "#336622")
grass_exclude_tolerance = st.sidebar.slider("Tolerancia Exclusión (Dist RGB)", 0, 150, 70, 5)
grass_color1_rgb_exclude = hex_to_rgb(grass_color1_hex) if enable_grass_exclusion else None
grass_color2_rgb_exclude = hex_to_rgb(grass_color2_hex) if enable_grass_exclusion else None

st.sidebar.subheader("Parámetros Posesión y Pases")
enable_possession = st.sidebar.checkbox("Calcular Posesión", value=True)
max_possession_distance = st.sidebar.slider("Distancia Máx. Posesión (px)", 10, 500, 150, 10)
highlight_player_in_possession = st.sidebar.checkbox("Resaltar jugador posesión", value=True)
enable_pass_counting = st.sidebar.checkbox("Contar Pases", value=True)
min_pass_distance = st.sidebar.slider("Distancia Mín. Pase (px)", 10, 300, 50, 5)

st.sidebar.subheader("Parámetros Kalman (Avanzado)")
kf_process_noise = st.sidebar.number_input("Ruido Proceso (Q)", 0.0, 1.0, 1e-3, 1e-4, format="%.4f")
kf_measurement_noise = st.sidebar.number_input("Ruido Medida (R)", 0.0, 10.0, 5e-1, 1e-1, format="%.2f")
frames_to_lose_track = st.sidebar.slider("Frames para perder seguimiento", 1, 60, 15)

st.sidebar.header("Cargar Video")
uploaded_file = st.sidebar.file_uploader("Elige archivo video", type=['mp4', 'mov', 'avi', 'mkv'])
# ------------------------------------

# --- Lógica Principal ---
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
    tfile.write(uploaded_file.read()); video_path = tfile.name
    st.info(f"Video '{uploaded_file.name}' cargado. Procesando...")
    cap = cv2.VideoCapture(video_path)
    total_processed_frames = 0 # Contador para el cálculo de posesión %
    total_original_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    st.info(f"Total frames: {total_original_frames}, FPS: {fps:.2f}")

    if not cap.isOpened(): st.error("Error al abrir video.")
    else:
        st_frame = st.empty(); progress_bar = st.progress(0); status_text = st.empty()
        start_time = time.time()
        kf = initialize_kalman()
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * kf_process_noise
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * kf_measurement_noise
        kalman_initialized = False; ball_estimated_pos = None; frames_since_ball_detected = 0

        # --- Contadores Posesión y Pases ---
        # Usar nombres de equipo actuales como claves
        possession_frames = {team_1_name: 0, team_2_name: 0}
        pass_counts = {team_1_name: 0, team_2_name: 0}
        previous_possession_info = {"team": None, "player_box": None, "ball_pos": None}
        # ------------------------------------

        while True:
            ball_detected_this_frame = False; detected_ball_box = None
            success, frame = cap.read()
            if not success:
                 status_text.success(f"Proceso completado ({total_processed_frames}/{total_original_frames} frames procesados)")
                 break # Fin del video
            total_processed_frames += 1 # Incrementar contador de frames procesados
            frame_bgr = frame.copy(); h_frame, w_frame, _ = frame_bgr.shape

            # A. Predicción Kalman
            if kalman_initialized:
                predicted_state = kf.predict()
                pred_x, pred_y = int(predicted_state[0, 0]), int(predicted_state[1, 0])
                ball_estimated_pos = (max(0, min(w_frame - 1, pred_x)), max(0, min(h_frame - 1, pred_y)))

            # B. Detección YOLO
            results = model.predict(frame_bgr, conf=confidence_threshold, verbose=False)
            result = results[0]

            # C. Procesar Detecciones
            ball_measurement = None; players_data = []; detected_boxes_data = []
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0]); current_box = [x1, y1, x2, y2]
                conf = box.conf[0]; cls_id = int(box.cls[0])
                cls_name = model.names.get(cls_id, f"ID_{cls_id}")
                is_player_or_gk = (PLAYER_CLS_ID is not None and cls_id == PLAYER_CLS_ID) or \
                                  (GOALKEEPER_CLS_ID is not None and cls_id == GOALKEEPER_CLS_ID)

                if BALL_CLS_ID is not None and cls_id == BALL_CLS_ID:
                    ball_detected_this_frame = True; frames_since_ball_detected = 0
                    detected_ball_box = current_box
                    center = calculate_center(current_box)
                    ball_measurement = np.array([[center[0]], [center[1]]], dtype=np.float32)
                    if not kalman_initialized:
                        kf.statePost = np.array([[center[0]], [center[1]], [0], [0]], dtype=np.float32)
                        kf.errorCovPost = np.eye(4, dtype=np.float32) * 10
                        kalman_initialized = True
                        ball_estimated_pos = (int(center[0]), int(center[1]))

                elif is_player_or_gk:
                    assigned_team_name = None; final_box_color = DEFAULT_BOX_COLOR_BGR
                    roi_y1, roi_y2 = max(0, y1), min(h_frame, y2); roi_x1, roi_x2 = max(0, x1), min(w_frame, x2)
                    if roi_y2 > roi_y1 and roi_x2 > roi_x1:
                        player_roi = frame_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
                        dominant_color_rgb = get_dominant_color(
                            player_roi, k=kmeans_k,
                            exclude_color1_rgb=grass_color1_rgb_exclude,
                            exclude_color2_rgb=grass_color2_rgb_exclude,
                            exclude_tolerance=grass_exclude_tolerance
                        )
                        if dominant_color_rgb:
                            dist_1 = color_distance(dominant_color_rgb, team_1_color_rgb)
                            dist_2 = color_distance(dominant_color_rgb, team_2_color_rgb)
                            if dist_1 < dist_2 and dist_1 < color_distance_threshold: assigned_team_name = team_1_name; final_box_color = TEAM_1_BOX_COLOR_BGR
                            elif dist_2 < dist_1 and dist_2 < color_distance_threshold: assigned_team_name = team_2_name; final_box_color = TEAM_2_BOX_COLOR_BGR
                            else: final_box_color = UNKNOWN_PLAYER_COLOR_BGR
                        else: final_box_color = UNKNOWN_PLAYER_COLOR_BGR
                    else: final_box_color = UNKNOWN_PLAYER_COLOR_BGR
                    player_center = calculate_center(current_box)
                    players_data.append({'box': current_box, 'center': player_center, 'team': assigned_team_name})
                    label = f"{cls_name}: {conf:.2f}";
                    if assigned_team_name: label = f"{assigned_team_name}: {conf:.2f}"
                    detected_boxes_data.append({'box': current_box, 'label': label, 'color': final_box_color, 'is_player': True})
                else: # Otros objetos
                    final_box_color = DEFAULT_BOX_COLOR_BGR
                    label = f"{cls_name}: {conf:.2f}"
                    detected_boxes_data.append({'box': current_box, 'label': label, 'color': final_box_color, 'is_player': False})

            # D. Actualización Kalman
            if kalman_initialized and ball_measurement is not None:
                corrected_state = kf.correct(ball_measurement)
                corr_x, corr_y = int(corrected_state[0, 0]), int(corrected_state[1, 0])
                ball_estimated_pos = (max(0, min(w_frame - 1, corr_x)), max(0, min(h_frame - 1, corr_y)))
            elif kalman_initialized:
                 frames_since_ball_detected += 1
                 if frames_since_ball_detected > frames_to_lose_track:
                     kalman_initialized = False; ball_estimated_pos = None

            # E. Calcular Posesión e Incrementar Contador
            possession_team_current = None; player_in_possession_box = None
            if enable_possession and kalman_initialized and ball_estimated_pos is not None and players_data:
                closest_player_info = None; min_dist = float('inf')
                for player in players_data:
                    if player['team'] is not None:
                        dist = math.dist(ball_estimated_pos, player['center'])
                        if dist < min_dist: min_dist = dist; closest_player_info = player
                if closest_player_info is not None and min_dist <= max_possession_distance:
                    possession_team_current = closest_player_info['team']
                    player_in_possession_box = closest_player_info['box']
                    # Usar get para evitar error si el nombre del equipo cambia a mitad
                    possession_frames[possession_team_current] = possession_frames.get(possession_team_current, 0) + 1

            if possession_team_current: possession_status = f"Pos: {possession_team_current}"
            elif enable_possession and kalman_initialized and ball_estimated_pos is not None: possession_status = "Pos: Balon Suelto"
            elif enable_possession and not kalman_initialized: possession_status = "Pos: Balon no seguido"
            else: possession_status = "Pos: Indet./Desact."

            # F. Detectar y Contar Pases por Equipo
            if enable_pass_counting and possession_team_current is not None and ball_estimated_pos is not None:
                if previous_possession_info["team"] == possession_team_current:
                    if previous_possession_info["player_box"] is not None and \
                       player_in_possession_box != previous_possession_info["player_box"]:
                        if previous_possession_info["ball_pos"] is not None:
                            ball_distance_moved = math.dist(ball_estimated_pos, previous_possession_info["ball_pos"])
                            if ball_distance_moved >= min_pass_distance:
                                pass_counts[possession_team_current] = pass_counts.get(possession_team_current, 0) + 1
                previous_possession_info = {"team": possession_team_current, "player_box": player_in_possession_box, "ball_pos": ball_estimated_pos}
            elif enable_pass_counting: previous_possession_info = {"team": None, "player_box": None, "ball_pos": None}

            # G. Dibujar Cajas y Balón
            for data in detected_boxes_data:
                box=data['box']; x1,y1,x2,y2=box; color=data['color']; label=data['label']; thickness=2
                if highlight_player_in_possession and player_in_possession_box is not None and data['is_player'] and box == player_in_possession_box: thickness = 4
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness)
                (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                ly1 = max(y1 - lh - bl, 0); ly2 = max(y1, ly1 + lh + bl)
                cv2.rectangle(frame_bgr, (x1, ly1), (x1 + lw, ly2), color, cv2.FILLED)
                tc = (0,0,0) if sum(color)>384 else (255,255,255)
                ty = max(y1 - bl + 3, lh); cv2.putText(frame_bgr, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tc, 1, cv2.LINE_AA)
            if ball_detected_this_frame and detected_ball_box is not None:
                x1, y1, x2, y2 = detected_ball_box
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), DETECTED_BALL_COLOR_BGR, 2)
            elif kalman_initialized and ball_estimated_pos is not None:
                cx, cy = ball_estimated_pos
                px1,py1 = max(0, cx-PREDICTED_BALL_BOX_SIZE_HALF), max(0, cy-PREDICTED_BALL_BOX_SIZE_HALF)
                px2,py2 = min(w_frame-1, cx+PREDICTED_BALL_BOX_SIZE_HALF), min(h_frame-1, cy+PREDICTED_BALL_BOX_SIZE_HALF)
                cv2.rectangle(frame_bgr, (px1, py1), (px2, py2), PREDICTED_BALL_COLOR_BGR, 2)

            # H. Dibujar Estado (Posesión %, Pases por Equipo)
            total_possession_frames_so_far = sum(possession_frames.values())
            percent_team1 = (possession_frames.get(team_1_name, 0) / total_possession_frames_so_far * 100) if total_possession_frames_so_far > 0 else 0
            percent_team2 = (possession_frames.get(team_2_name, 0) / total_possession_frames_so_far * 100) if total_possession_frames_so_far > 0 else 0
            pass_text = ""
            if enable_pass_counting:
                t1_short = team_1_name.split()[0][:5]; t2_short = team_2_name.split()[0][:5]
                pass_text = f" | P({t1_short}):{pass_counts.get(team_1_name, 0)} P({t2_short}):{pass_counts.get(team_2_name, 0)}"
            display_text = f"{possession_status} ({percent_team1:.0f}%/{percent_team2:.0f}%){pass_text}"
            pos_text_x=15; pos_text_y=40; pos_font_scale=0.8; pos_font_thickness=2
            pos_text_color=(255, 255, 255); pos_bg_color=(0, 0, 0); pos_padding=5
            (w_text, h_text), bl_text = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, pos_font_scale, pos_font_thickness)
            bg_x1=max(0, pos_text_x - pos_padding); bg_y1=max(0, pos_text_y - h_text - bl_text - pos_padding)
            bg_x2=min(w_frame, pos_text_x + w_text + pos_padding); bg_y2=min(h_frame, pos_text_y + bl_text + pos_padding)
            if bg_y2 > bg_y1 and bg_x2 > bg_x1: cv2.rectangle(frame_bgr, (bg_x1, bg_y1), (bg_x2, bg_y2), pos_bg_color, cv2.FILLED)
            cv2.putText(frame_bgr, display_text, (pos_text_x, pos_text_y), cv2.FONT_HERSHEY_SIMPLEX, pos_font_scale, pos_text_color, pos_font_thickness, cv2.LINE_AA)

            # I. Mostrar Frame y Progreso
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, caption=f"Frame {total_processed_frames}/{total_original_frames}", channels="RGB", use_container_width=True)
            # Usar total_processed_frames para la barra de progreso relativa al total original
            progress_value = total_processed_frames / total_original_frames if total_original_frames > 0 else 0
            progress_bar.progress(min(progress_value, 1.0)) # Asegurar que no exceda 1.0
            elapsed_time = time.time() - start_time
            eta = (elapsed_time / total_processed_frames) * (total_original_frames - total_processed_frames) if total_processed_frames > 0 and total_processed_frames < total_original_frames else 0
            status_text.info(f"Procesando frame {total_processed_frames}/{total_original_frames}... ETA: {eta:.0f}s")

        # --- Limpieza Final ---
        cap.release(); tfile.close()
        try: os.remove(video_path)
        except Exception as e: st.warning(f"No se pudo eliminar tmp file: {e}")

        # --- Mostrar Resumen Final ---
        st.subheader("Resumen del Video Procesado")
        # Recalcular porcentajes finales basados en el total de frames con posesión
        total_possession_frames_final = sum(possession_frames.values())
        final_percent_team1 = (possession_frames.get(team_1_name, 0) / total_possession_frames_final * 100) if total_possession_frames_final > 0 else 0
        final_percent_team2 = (possession_frames.get(team_2_name, 0) / total_possession_frames_final * 100) if total_possession_frames_final > 0 else 0

        col1, col2 = st.columns(2)
        with col1:
             st.metric(label=f"Posesión {team_1_name}", value=f"{final_percent_team1:.1f}%")
             st.metric(label=f"Pases {team_1_name}", value=pass_counts.get(team_1_name, 0))
        with col2:
             st.metric(label=f"Posesión {team_2_name}", value=f"{final_percent_team2:.1f}%")
             st.metric(label=f"Pases {team_2_name}", value=pass_counts.get(team_2_name, 0))
        st.info(f"Total de frames con posesión registrada: {total_possession_frames_final}")
        # ---------------------------

else:
    st.info("Esperando a que subas un archivo de video.")

st.sidebar.info("Streamlit + YOLOv8 + Full Features")
# -----------------------------------------------------------