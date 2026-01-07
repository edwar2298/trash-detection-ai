from ultralytics import YOLO
import cv2
import torch
import time

# -----------------------
# 1. DICCIONARIO DE TRADUCCIÓN (ESPAÑOL -> INGLÉS)
# -----------------------
# Aquí defines cómo quieres que se vea cada clase en la pantalla
TRADUCCION = {
    "persona": "PERSON",
    "person": "PERSON",
    "basura": "TRASH",
    "trash": "TRASH",
    "placa": "LICENSE PLATE",
    "license_plate": "LICENSE PLATE",
    "tirar": "THROWING TRASH",
    "accion": "ACTION",
    "accion_tirar": "THROWING TRASH"
}

# -----------------------
# 2. CARGAR MODELOS
# -----------------------
print("⏳ Loading Models (Hybrid System)...")
model_v8 = YOLO("models/best_yolov8.pt")
model_world = YOLO("models/best_yoloworld.pt")

names_v8 = model_v8.names
names_world = model_world.names

# -----------------------
# 3. CAPTURA DE VIDEO
# -----------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("❌ Error: Could not open camera")
    exit()

prev_time = 0

print("🚀 Hybrid Detection System Started!")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS Calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if current_time != prev_time else 0
    prev_time = current_time

    # -----------------------
    # PREDICCIONES (INFERENCIA)
    # -----------------------
    res_v8 = model_v8(frame, verbose=False)[0]
    res_world = model_world(frame, verbose=False)[0]

    # -----------------------
    # FUSIÓN DE DATOS
    # -----------------------
    # Extraemos datos de YOLOv8
    boxes_v8 = res_v8.boxes.xyxy
    scores_v8 = res_v8.boxes.conf
    cls_v8 = res_v8.boxes.cls

    # Extraemos datos de YOLO-World
    boxes_world = res_world.boxes.xyxy
    scores_world = res_world.boxes.conf
    cls_world = res_world.boxes.cls

    # Concatenamos (Unimos) las detecciones de ambos modelos
    boxes_comb = torch.cat((boxes_v8, boxes_world), dim=0)
    scores_comb = torch.cat((scores_v8, scores_world), dim=0)
    cls_comb = torch.cat((cls_v8, cls_world), dim=0)

    # Guardamos cuántas eran de v8 para saber cuál es cuál
    count_v8 = len(cls_v8)

    annotated = frame.copy()

    # -----------------------
    # DIBUJAR RESULTADOS
    # -----------------------
    for i in range(len(boxes_comb)):
        x1, y1, x2, y2 = boxes_comb[i].int().tolist()
        score = float(scores_comb[i])
        cls = int(cls_comb[i])

        # A. IDENTIFICAR MODELO Y OBTENER NOMBRE ORIGINAL
        if i < count_v8:
            # Viene de YOLOv8
            original_label = names_v8[cls] 
            color = (0, 255, 0) # Verde para v8
            model_tag = "v8"
        else:
            # Viene de YOLO-World
            original_label = names_world[cls]
            color = (255, 255, 0) # Cyan/Amarillo para World
            model_tag = "World"

        # B. TRADUCIR AL INGLÉS
        # Busca el nombre en el diccionario, si no lo encuentra, lo pone en mayúsculas por defecto
        english_label = TRADUCCION.get(original_label.lower(), original_label.upper())

        # Texto final: "PERSON v8 0.85"
        text = f"{english_label} ({model_tag}) {score:.2f}"

        # C. PINTAR EN PANTALLA
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Fondo pequeño para el texto (Mejora lectura)
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - 25), (x1 + w, y1), color, -1)
        
        cv2.putText(
            annotated,
            text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0), # Texto negro para contraste
            2
        )

    # Mostrar FPS
    cv2.putText(annotated, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Leyenda visual (Opcional, ayuda a entender qué color es qué modelo)
    cv2.putText(annotated, "Green: YOLOv8", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(annotated, "Cyan: YOLO-World", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("YOLO-HCLT Hybrid System (English)", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()