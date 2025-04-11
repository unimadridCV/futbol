import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import os

# --- Configuración ---
RUTA_MODELO_DETECCION = "modelo_deteccion_futbol.pt"
RUTA_MODELO_PUNTOS_CLAVE = "puntos_clave_futbol.pt"
RUTA_IMAGEN = "imagen_futbol.jpg"

# Dimensiones del campo en metros
ANCHO_CAMPO_METROS = 100.0
ALTO_CAMPO_METROS = 60.0

# Mapeo de clases del modelo de detección
MAPEO_CLASES_DETECCION = {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}

# --- Validación de Archivos ---
if not os.path.exists(RUTA_MODELO_DETECCION):
    print(f"Error: No se encontró el modelo de detección en {RUTA_MODELO_DETECCION}")
    exit()
if not os.path.exists(RUTA_MODELO_PUNTOS_CLAVE):
    print(f"Error: No se encontró el modelo de puntos clave en {RUTA_MODELO_PUNTOS_CLAVE}")
    exit()
if not os.path.exists(RUTA_IMAGEN):
    print(f"Error: No se encontró la imagen en {RUTA_IMAGEN}")
    exit()

# --- Cargar Modelos ---
try:
    modelo_deteccion = YOLO(RUTA_MODELO_DETECCION)
    modelo_puntos_clave = YOLO(RUTA_MODELO_PUNTOS_CLAVE)
    print("Modelos cargados exitosamente.")
except Exception as e:
    print(f"Error al cargar los modelos: {e}")
    exit()

# --- Cargar Imagen ---
imagen = cv2.imread(RUTA_IMAGEN)
if imagen is None:
    print(f"Error al cargar la imagen: {RUTA_IMAGEN}")
    exit()
print(f"Imagen cargada: {RUTA_IMAGEN} (Dimensiones: {imagen.shape[1]}x{imagen.shape[0]})")

# --- 1. Detectar Puntos Clave del Campo ---
print("Detectando puntos clave del campo...")
results_puntos_clave = modelo_puntos_clave(imagen, verbose=False) # verbose=False para menos output

# Extraer coordenadas de los puntos clave detectados
# Asumimos que el modelo devuelve los puntos clave en el primer resultado
pts_imagen_pixeles = None
if results_puntos_clave and results_puntos_clave[0].keypoints is not None:
    # Tomamos las coordenadas xy del primer conjunto de keypoints detectado
    keypoints_data = results_puntos_clave[0].keypoints.xy.cpu().numpy()
    if keypoints_data.shape[0] > 0:
         # Asumimos que la primera detección es la relevante si hay múltiples
        pts_imagen_pixeles = keypoints_data[0]
        print(f"Detectados {len(pts_imagen_pixeles)} puntos clave en la imagen.")
    else:
        print("Advertencia: El modelo de puntos clave se ejecutó pero no detectó puntos.")
else:
    print("Advertencia: No se obtuvieron resultados del modelo de puntos clave.")

# --- 2. Calcular Homografía (si se detectaron suficientes puntos) ---
matriz_homografia = None
if pts_imagen_pixeles is not None and len(pts_imagen_pixeles) >= 4:
    # **¡IMPORTANTE!** Define aquí las coordenadas REALES (en metros)
    # correspondientes a los PRIMEROS 4 PUNTOS CLAVE detectados por tu modelo.
    # Esta es la suposición más crítica. Ajusta según tu modelo.
    # Ejemplo: Asumiendo que detecta las 4 esquinas en orden:
    # (0,0) Top-Left, (ANCHO, 0) Top-Right, (ANCHO, ALTO) Bottom-Right, (0, ALTO) Bottom-Left
    pts_campo_real_metros = np.array([
        [0.0, 0.0],                      # Punto correspondiente al primer keypoint detectado
        [ANCHO_CAMPO_METROS, 0.0],       # Punto correspondiente al segundo keypoint detectado
        [ANCHO_CAMPO_METROS, ALTO_CAMPO_METROS], # ... tercero ...
        [0.0, ALTO_CAMPO_METROS]         # ... cuarto ...
    ], dtype=np.float32)

    # Aseguramos usar solo los primeros 4 puntos para findHomography
    pts_imagen_pixeles_4 = pts_imagen_pixeles[:4]

    print("Puntos en imagen (primeros 4):")
    print(pts_imagen_pixeles_4)
    print("Puntos en campo real (metros):")
    print(pts_campo_real_metros)

    # Calcular la matriz de homografía
    matriz_homografia, status = cv2.findHomography(pts_imagen_pixeles_4, pts_campo_real_metros)

    if matriz_homografia is not None:
        print("Matriz de Homografía calculada exitosamente.")
    else:
        print("Error: No se pudo calcular la matriz de homografía.")
        print("Verifica que los puntos clave detectados sean correctos y no colineales.")
else:
    print("Advertencia: No se detectaron suficientes puntos clave (se necesitan >= 4) para calcular la homografía.")
    print("No se podrá estimar la posición real de los jugadores.")

# --- 3. Detectar Jugadores, Porteros y Árbitros ---
print("Detectando jugadores, porteros y árbitros...")
results_deteccion = modelo_deteccion(imagen, verbose=False)

# --- 4. Estimar Posición y Crear Tabla ---
datos_posiciones = []

# Procesar detecciones solo si tenemos la homografía
if matriz_homografia is not None:
    print("Estimando posiciones en el campo...")
    for box in results_deteccion[0].boxes:
        clase_id = int(box.cls[0])
        nombre_clase = MAPEO_CLASES_DETECCION.get(clase_id, 'Desconocido')

        # Nos interesan solo las personas (jugador, portero, árbitro)
        if nombre_clase in ['goalkeeper', 'player', 'referee']:
            # Obtener coordenadas del bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            # Calcular el punto base del jugador (centro inferior del bounding box)
            # Este punto suele representar mejor la posición en el campo (pies)
            punto_base_imagen = np.array([[(x1 + x2) / 2, y2]], dtype=np.float32)

            # Aplicar la transformación de perspectiva (homografía)
            # cv2.perspectiveTransform necesita una forma específica (N, 1, 2)
            punto_base_real = cv2.perspectiveTransform(punto_base_imagen.reshape(-1, 1, 2), matriz_homografia)

            if punto_base_real is not None:
                # Extraer coordenadas reales (X, Y) en metros
                x_real_m = punto_base_real[0][0][0]
                y_real_m = punto_base_real[0][0][1]

                # Añadir a la lista de resultados
                datos_posiciones.append({
                    'Clase': nombre_clase,
                    'ID_Clase': clase_id,
                    'Confianza': float(box.conf[0]),
                    'X_Imagen (px)': int(punto_base_imagen[0, 0]),
                    'Y_Imagen (px)': int(punto_base_imagen[0, 1]),
                    'X_Campo (m)': round(x_real_m, 2),
                    'Y_Campo (m)': round(y_real_m, 2)
                })
            else:
                 print(f"Advertencia: No se pudo transformar el punto para un {nombre_clase}")

else:
    print("No se puede estimar la posición real sin la matriz de homografía.")
    # Opcionalmente, podrías llenar la tabla solo con detecciones en píxeles
    for box in results_deteccion[0].boxes:
        clase_id = int(box.cls[0])
        nombre_clase = MAPEO_CLASES_DETECCION.get(clase_id, 'Desconocido')
        if nombre_clase in ['goalkeeper', 'player', 'referee']:
             x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
             datos_posiciones.append({
                    'Clase': nombre_clase,
                    'ID_Clase': clase_id,
                    'Confianza': float(box.conf[0]),
                    'X_Imagen (px)': int((x1+x2)/2),
                    'Y_Imagen (px)': y2,
                    'X_Campo (m)': 'N/A',
                    'Y_Campo (m)': 'N/A'
                })


# --- 5. Mostrar Resultados en una Tabla ---
if datos_posiciones:
    df_posiciones = pd.DataFrame(datos_posiciones)
    # Ordenar opcionalmente, por ejemplo por clase y luego por X
    df_posiciones = df_posiciones.sort_values(by=['Clase', 'X_Campo (m)'])

    print("\n--- Tabla de Posiciones Estimadas ---")
    # Usar to_string() para mostrar la tabla completa sin truncar
    print(df_posiciones.to_string(index=False))
else:
    print("\nNo se detectaron jugadores, porteros o árbitros, o no se pudo calcular la posición.")


# --- 6. (Opcional) Visualización ---
print("\nGenerando imagen con visualizaciones (opcional)...")
imagen_visualizacion = imagen.copy()

# Dibujar puntos clave detectados
if pts_imagen_pixeles is not None:
    for i, pt in enumerate(pts_imagen_pixeles):
        cv2.circle(imagen_visualizacion, tuple(map(int, pt)), 7, (0, 255, 0), -1) # Verde
        cv2.putText(imagen_visualizacion, str(i), tuple(map(int, pt - np.array([10, 10]))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


# Dibujar bounding boxes y posiciones estimadas
for i, row in enumerate(datos_posiciones):
    nombre_clase = row['Clase']
    conf = row['Confianza']
    x_img, y_img = row['X_Imagen (px)'], row['Y_Imagen (px)']
    x_real, y_real = row['X_Campo (m)'], row['Y_Campo (m)']

    # Necesitamos recuperar el bbox original (no está guardado directamente)
    # Volvemos a buscar la detección por su punto base (puede no ser perfecto si hay solapamiento)
    # Alternativa: guardar bbox en `datos_posiciones` si se necesita dibujar siempre
    box_encontrado = None
    min_dist = float('inf')
    current_box = None

    # Esta parte es ineficiente, sería mejor guardar el bbox en `datos_posiciones`
    for box in results_deteccion[0].boxes:
         clase_id_box = int(box.cls[0])
         nombre_clase_box = MAPEO_CLASES_DETECCION.get(clase_id_box, 'Desconocido')
         if nombre_clase_box == nombre_clase:
            x1b, y1b, x2b, y2b = map(int, box.xyxy[0].cpu().numpy())
            cx_b = (x1b + x2b) / 2
            cy_b = y2b # Punto base Y
            dist = np.sqrt((cx_b - x_img)**2 + (cy_b - y_img)**2)
            # Si el punto base coincide (o está muy cerca)
            if dist < min_dist and abs(float(box.conf[0]) - conf) < 0.01 : # Comprobar confianza también
                 min_dist = dist
                 current_box = (x1b, y1b, x2b, y2b)


    if current_box is not None and min_dist < 5: # Umbral pequeño para asegurar que es el mismo
        x1, y1, x2, y2 = current_box
        color = (255, 0, 0) # Azul por defecto
        if nombre_clase == 'goalkeeper': color = (0, 255, 255) # Amarillo
        if nombre_clase == 'referee': color = (0, 0, 255) # Rojo

        # Dibujar Bounding Box
        cv2.rectangle(imagen_visualizacion, (x1, y1), (x2, y2), color, 2)

        # Etiqueta con clase y posición real (si está disponible)
        etiqueta = f"{nombre_clase}"
        if x_real != 'N/A':
             etiqueta += f" ({x_real}m, {y_real}m)"

        # Poner texto encima del bbox
        (w, h), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y1_label = max(y1, h + 5) # Asegurar que no se salga por arriba
        cv2.rectangle(imagen_visualizacion, (x1, y1_label - h - 5), (x1 + w, y1_label), color, -1)
        cv2.putText(imagen_visualizacion, etiqueta, (x1, y1_label - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        # Si no se pudo recuperar el bbox, al menos dibujar el punto base estimado
        cv2.circle(imagen_visualizacion, (x_img, y_img), 5, (0,0,255), -1)


# Guardar o mostrar la imagen resultante
ruta_salida = "resultado_deteccion_posicion.jpg"
cv2.imwrite(ruta_salida, imagen_visualizacion)
print(f"\nImagen con visualizaciones guardada en: {ruta_salida}")

# Mostrar imagen (opcional, requiere entorno gráfico)
# cv2.imshow("Detecciones y Posiciones Estimadas", imagen_visualizacion)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print("\nScript finalizado.")