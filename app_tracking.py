import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import tempfile
import os
import time
import traceback

# --- Configuración ---
FIELD_WIDTH_M = 100.0
FIELD_HEIGHT_M = 60.0
DETECTION_MODEL_PATH = "modelo_deteccion_futbol.pt"
KEYPOINT_MODEL_PATH = "puntos_clave_futbol.pt"
TRACKER_CONFIG = "bytetrack.yaml" # o "botsort.yaml"
KEYPOINT_INDICES_FOR_HOMOGRAPHY = [0, 1, 2, 3]
DESTINATION_POINTS = np.array([
    [0, 0],
    [FIELD_WIDTH_M, 0],
    [FIELD_WIDTH_M, FIELD_HEIGHT_M],
    [0, FIELD_HEIGHT_M]
], dtype=np.float32)
PLAYER_CLASSES_IDS = [1, 2] # IDs de clase para 'goalkeeper' y 'player'
CLASS_NAMES = {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}

# --- Carga de Modelos (Cacheado para eficiencia) ---
@st.cache_resource
def load_detection_model(path):
    """Carga el modelo de detección/seguimiento YOLO."""
    try:
        model = YOLO(path)
        # Comprobar si los nombres están en el modelo, pero usar nuestro diccionario CLASS_NAMES
        # st.info(f"Modelo cargó nombres: {model.names}") # Opcional para depurar
        st.success(f"Modelo de detección cargado desde {path}")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo de detección ({path}): {e}")
        st.error(traceback.format_exc())
        return None

@st.cache_resource
def load_keypoint_model(path):
    """Carga el modelo de detección de puntos clave YOLO."""
    try:
        model = YOLO(path)
        st.success(f"Modelo de puntos clave cargado desde {path}")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo de puntos clave ({path}): {e}")
        st.error(traceback.format_exc())
        return None

# --- Funciones Auxiliares ---
def draw_detections(frame, results_track):
    """Dibuja las cajas delimitadoras y los IDs de seguimiento en el frame."""
    if results_track and results_track[0].boxes is not None and results_track[0].boxes.id is not None:
        boxes = results_track[0].boxes.xyxy.cpu().numpy()
        track_ids = results_track[0].boxes.id.int().cpu().tolist()
        confs = results_track[0].boxes.conf.cpu().numpy()
        clss = results_track[0].boxes.cls.cpu().numpy()

        for box, track_id, conf, cls_id in zip(boxes, track_ids, confs, clss):
            x1, y1, x2, y2 = map(int, box)
            class_name = CLASS_NAMES.get(int(cls_id), 'Desconocido')
            label = f"ID:{track_id} {class_name} {conf:.2f}"
            color = (0, 255, 0) if int(cls_id) in PLAYER_CLASSES_IDS else (255, 0, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Opcional: Dibujar el punto usado para mapeo (inferior central)
            if int(cls_id) in PLAYER_CLASSES_IDS:
                center_bottom_x = int((x1 + x2) / 2)
                center_bottom_y = int(y2)
                cv2.circle(frame, (center_bottom_x, center_bottom_y), 5, (255, 255, 0), -1) # Círculo amarillo

    return frame

def draw_keypoints(frame, results_kpts):
    """Dibuja los puntos clave detectados en el frame."""
    if results_kpts and results_kpts[0].keypoints is not None:
        keypoints_data = results_kpts[0].keypoints.xy.cpu().numpy()
        if keypoints_data.size > 0:
            for i, kpt_set in enumerate(keypoints_data):
                 if i == 0:
                    for idx, (x, y) in enumerate(kpt_set):
                        if x > 0 and y > 0:
                            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                            cv2.putText(frame, str(idx), (int(x)+5, int(y)+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return frame

def calculate_homography(results_kpts, src_indices, dst_points):
    """Calcula la matriz de homografía si se detectan suficientes puntos clave."""
    homography_matrix = None
    detected_src_pts = []
    if results_kpts and results_kpts[0].keypoints is not None:
        keypoints_data = results_kpts[0].keypoints.xy.cpu().numpy()
        if keypoints_data.size > 0 and len(keypoints_data[0]) >= max(src_indices) + 1:
            all_kpts = keypoints_data[0]
            valid_points_count = 0
            src_pts_list = []
            for idx in src_indices:
                pt = all_kpts[idx]
                if pt[0] > 0 and pt[1] > 0:
                    src_pts_list.append(pt)
                    valid_points_count += 1
            if valid_points_count == len(src_indices):
                src_pts_np = np.array(src_pts_list, dtype=np.float32)
                detected_src_pts = src_pts_np
                homography_matrix, mask = cv2.findHomography(src_pts_np, dst_points, cv2.RANSAC, 5.0)
                if homography_matrix is None:
                    # st.warning("findHomography falló.") # Descomentar para depurar
                    detected_src_pts = []
            # else:
                 # st.warning(f"Puntos válidos: {valid_points_count}/{len(src_indices)}") # Descomentar para depurar
    return homography_matrix, detected_src_pts

# MODIFICADO: Ahora devuelve una lista de diccionarios con más detalles
def get_player_positions_details(results_track, homography_matrix):
    """
    Obtiene detalles de posición (ID, Pixel X/Y, Campo X/Y) para jugadores detectados.
    Devuelve una lista de diccionarios, uno por jugador mapeado en este frame.
    """
    player_details_list = []

    if homography_matrix is None or results_track is None or results_track[0].boxes is None or results_track[0].boxes.id is None:
        return player_details_list # No se puede mapear

    boxes = results_track[0].boxes.xyxy.cpu().numpy()
    track_ids = results_track[0].boxes.id.int().cpu().tolist()
    clss = results_track[0].boxes.cls.cpu().numpy()

    for box, track_id, cls_id in zip(boxes, track_ids, clss):
        if int(cls_id) in PLAYER_CLASSES_IDS:
            x1, y1, x2, y2 = box
            # Punto inferior central en píxeles
            pixel_x = (x1 + x2) / 2
            pixel_y = y2
            player_point_pixel = np.array([[(pixel_x, pixel_y)]], dtype=np.float32)

            # Aplicar la transformación de perspectiva
            transformed_point = cv2.perspectiveTransform(player_point_pixel, homography_matrix)

            if transformed_point is not None and transformed_point.size > 0:
                map_x, map_y = transformed_point[0][0]
                # Asegurar límites
                map_x = max(0.0, min(FIELD_WIDTH_M, map_x))
                map_y = max(0.0, min(FIELD_HEIGHT_M, map_y))

                player_details_list.append({
                    "ID Jugador": track_id,
                    "Pixel X": int(pixel_x), # Coordenada X en la imagen
                    "Pixel Y": int(pixel_y), # Coordenada Y en la imagen
                    "Campo X (m)": round(map_x, 2), # Coordenada X en metros
                    "Campo Y (m)": round(map_y, 2)  # Coordenada Y en metros
                })

    return player_details_list

# --- Interfaz de Streamlit ---
st.set_page_config(layout="wide", page_title="Análisis de Fútbol")
st.title("⚽ Análisis de Video de Fútbol: Detección y Posicionamiento 🥅")

# Cargar modelos al inicio
detection_model = load_detection_model(DETECTION_MODEL_PATH)
keypoint_model = load_keypoint_model(KEYPOINT_MODEL_PATH)

# Subida de archivo de video
uploaded_file = st.file_uploader("Sube un archivo de video (MP4, AVI, MOV, MKV)", type=["mp4", "avi", "mov", "mkv"])

# Contenedores para la salida
col1, col2 = st.columns([3, 1])
with col1:
    stframe = st.empty()
with col2:
    st.subheader("Posiciones en Vivo") # Añadir título a la tabla
    sttable = st.empty()
status_text = st.empty()

if uploaded_file is not None and detection_model is not None and keypoint_model is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        video_path = tmp_file.name

    status_text.info(f"Archivo '{uploaded_file.name}' cargado. Iniciando procesamiento...")

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error al abrir el archivo de video.")
            if os.path.exists(video_path):
                 os.unlink(video_path)
        else:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            st.sidebar.text(f"Resolución: {frame_width}x{frame_height}")
            st.sidebar.text(f"FPS: {fps:.2f}")
            st.sidebar.text(f"Frames Totales: {total_frames}")

            progress_bar = st.progress(0)
            frame_count = 0
            # Sigue siendo útil para el resumen final
            last_positions_mapped = {}
            homography_status = "Calculando..."

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    status_text.success("Procesamiento completado o fin del video alcanzado.")
                    break

                frame_count += 1
                start_time = time.time()

                # 1. Detectar Puntos Clave
                results_kpts = keypoint_model.predict(frame, verbose=False, conf=0.6) # Ajusta conf si es necesario

                # 2. Calcular Homografía
                H, used_kpts = calculate_homography(results_kpts, KEYPOINT_INDICES_FOR_HOMOGRAPHY, DESTINATION_POINTS)
                homography_status = "✅ Calculada" if H is not None else "❌ No calculada"

                # 3. Detectar y Rastrear Objetos
                results_track = detection_model.track(frame, persist=True, tracker=TRACKER_CONFIG, verbose=False, conf=0.5, classes=PLAYER_CLASSES_IDS + [0, 3]) # Ajusta conf, detecta todo pero mapea solo jugadores

                # 4. OBTENER DETALLES DE POSICIÓN (NUEVA FUNCIÓN)
                current_frame_player_details = []
                if H is not None:
                    current_frame_player_details = get_player_positions_details(results_track, H)

                    # Actualizar last_positions_mapped para el resumen final
                    for player_detail in current_frame_player_details:
                        last_positions_mapped[player_detail["ID Jugador"]] = (
                            player_detail["Campo X (m)"],
                            player_detail["Campo Y (m)"]
                        )

                # 5. Preparar frame para mostrar
                processed_frame = frame.copy()
                processed_frame = draw_keypoints(processed_frame, results_kpts)
                processed_frame = draw_detections(processed_frame, results_track)

                # Mostrar frame
                stframe.image(processed_frame, channels="BGR", caption=f"Frame {frame_count}/{total_frames}")

                # 6. ACTUALIZAR TABLA EN VIVO
                if current_frame_player_details:
                     # Crear DataFrame directamente de la lista de diccionarios
                     df_live = pd.DataFrame(current_frame_player_details)
                     # Reordenar columnas si se desea
                     df_live = df_live[["ID Jugador", "Pixel X", "Pixel Y", "Campo X (m)", "Campo Y (m)"]]
                     sttable.dataframe(df_live, height=300) # Ajusta height según necesites
                elif H is None:
                    sttable.warning("Homografía no calculada. No se pueden mapear posiciones.")
                else:
                    sttable.info("No hay jugadores (clase 1 o 2) detectados o mapeados en este frame.")

                # Actualizar progreso
                progress_percent = frame_count / total_frames if total_frames > 0 else 0
                progress_bar.progress(progress_percent)
                # end_time = time.time() # Descomentar si quieres ver tiempo por frame
                # processing_time = end_time - start_time
                # status_text.info(f"Procesando: Frame {frame_count}/{total_frames} | H: {homography_status} | T: {processing_time:.3f}s")


            cap.release()
            progress_bar.empty()
            status_text.success("¡Procesamiento de video finalizado!")

            # Mostrar tabla final consolidada (solo mapeadas)
            st.sidebar.subheader("Posiciones Finales Consolidadas")
            if last_positions_mapped:
                 final_position_data = [{"ID Jugador": pid, "X (m)": pos[0], "Y (m)": pos[1]}
                                         for pid, pos in sorted(last_positions_mapped.items())]
                 final_df = pd.DataFrame(final_position_data)
                 st.sidebar.dataframe(final_df)
            else:
                 st.sidebar.info("No se registraron posiciones mapeadas de jugadores.")

    except Exception as e:
        st.error("Ocurrió un error inesperado durante el procesamiento:")
        st.error(e)
        st.error(traceback.format_exc())

    finally:
        if 'video_path' in locals() and os.path.exists(video_path):
            try:
                os.unlink(video_path)
            except PermissionError:
                 st.warning(f"No se pudo eliminar el archivo temporal {video_path}.")

elif uploaded_file is None:
    st.info("👈 Por favor, sube un archivo de video.")
else: # Error en carga de modelos
    st.error("Uno o más modelos no se cargaron correctamente. Verifica las rutas y los archivos.")