from ultralytics import YOLO
import cv2
import time
import torch
import easyocr
import re
import numpy as np
import mysql.connector
import os
from datetime import datetime

# -----------------------
# 1. DIRECTORIOS DE EVIDENCIA
# -----------------------
if not os.path.exists('evidencias'): os.makedirs('evidencias')
if not os.path.exists('evidencias/videos'): os.makedirs('evidencias/videos')
if not os.path.exists('evidencias/placas'): os.makedirs('evidencias/placas') 

# -----------------------
# 2. BASE DE DATOS
# -----------------------
def buscar_propietario(placa_buscada):
    try:
        conexion = mysql.connector.connect(host="localhost", user="root", password="", database="sistema_placas")
        cursor = conexion.cursor()
        cursor.execute("SELECT nombre, direccion, celular FROM propietarios WHERE placa = %s", (placa_buscada,))
        resultado = cursor.fetchone()
        conexion.close()
        if resultado:
            return {"nombre": resultado[0], "direccion": resultado[1], "celular": resultado[2], "estado": "REGISTRADO"}
        else:
            return {"estado": "NO REGISTRADO"}
    except Exception:
        return {"estado": "ERROR BD"}

# -----------------------
# 3. INICIALIZACIÓN
# -----------------------
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"🚀 SECURITY SYSTEM ACTIVE... GPU: {device.upper()}")

reader = easyocr.Reader(['en'], gpu=(device == 'cuda:0'))
model = YOLO("models/best_yoloworld.pt")
model.to(device)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# -----------------------
# VARIABLES DE CONTROL
# -----------------------
ultima_placa_validada = ""
datos_propietario_actual = None
ultimo_tiempo_lectura = 0

GRABANDO = False
inicio_grabacion = 0
duracion_grabacion = 10 
out = None 

cv2.namedWindow("Sistema Policial", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Sistema Policial", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    current_time = time.time()

    # ============================================================
    # MODO 1: GRABACIÓN (RECORDING MODE)
    # ============================================================
    if GRABANDO:
        tiempo_transcurrido = current_time - inicio_grabacion
        tiempo_restante = duracion_grabacion - tiempo_transcurrido

        if out: out.write(frame)

        # Interfaz de Grabación
        cv2.circle(frame, (50, 50), 20, (0, 0, 255), -1)
        cv2.putText(frame, f"RECORDING EVIDENCE: {int(tiempo_restante)}s", (80, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "DETECTION PAUSED - PROTOCOL ACTIVE", (80, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if tiempo_transcurrido >= duracion_grabacion:
            GRABANDO = False
            if out: out.release()
            print("✅ Evidence saved. Resuming surveillance.")

    # ============================================================
    # MODO 2: VIGILANCIA (SURVEILLANCE MODE)
    # ============================================================
    else:
        results = model.predict(frame, device=device, half=True, conf=0.30, iou=0.50, imgsz=1280, verbose=False, max_det=15)
        det_boxes = results[0].boxes
        
        accion_tirar_detectada = False
        coords_placa_actual = None 

        for box in det_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id] # Nombre original del modelo (ej: "persona")

            # -----------------------------------------------
            # TRADUCCIÓN A INGLÉS (DISPLAY LABEL)
            # -----------------------------------------------
            label_display = class_name.upper() # Por defecto
            
            if class_name in ['persona', 'person']: label_display = "PERSON"
            elif class_name in ['basura', 'trash']: label_display = "TRASH"
            elif class_name in ['placa', 'placas']: label_display = "LICENSE PLATE"
            elif class_name in ['tirar', 'accion', 'accion_tirar', 'throwing']: label_display = "THROWING TRASH"

            # -----------------------
            # COLORES (Usamos el nombre original para lógica)
            # -----------------------
            color = (0, 255, 0) 
            if class_name in ['persona', 'person']: color = (255, 0, 0)     
            elif class_name in ['basura', 'trash']: color = (0, 0, 255)     
            elif class_name in ['placa', 'placas']: 
                color = (0, 255, 255) 
                coords_placa_actual = (x1, y1, x2, y2)
            
            elif class_name in ['tirar', 'accion', 'accion_tirar', 'throwing']: 
                color = (0, 165, 255) 
                accion_tirar_detectada = True

            # DIBUJAR CAJAS Y ETIQUETAS (EN INGLÉS)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Solo escribimos el nombre si NO es placa (la placa lleva su propio panel abajo)
            if class_name not in ['placa', 'placas']:
                cv2.putText(frame, label_display, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # -----------------------
            # LÓGICA OCR (LECTURA DE PLACA)
            # -----------------------
            if class_name in ['placa', 'placas'] and (current_time - ultimo_tiempo_lectura > 0.5):
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    try:
                        roi_zoom = cv2.resize(roi, None, fx=3, fy=3)
                        gray = cv2.cvtColor(roi_zoom, cv2.COLOR_BGR2GRAY)
                        ocr_res = reader.readtext(gray, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                        
                        if ocr_res:
                            txt = "".join(ocr_res).upper().replace("PERU","").replace("PE","")
                            txt = re.sub(r'[^A-Z0-9]', '', txt)
                            if len(txt) > 6: txt = txt[:6]
                            
                            if len(txt) == 6:
                                placa_fmt = f"{txt[:3]}-{txt[3:]}"
                                if placa_fmt != ultima_placa_validada:
                                    ultima_placa_validada = placa_fmt
                                    datos_propietario_actual = buscar_propietario(placa_fmt)
                                ultimo_tiempo_lectura = current_time
                    except: pass

        # -----------------------
        # PANEL DE INFORMACIÓN (MODO INGLÉS)
        # -----------------------
        if datos_propietario_actual:
            cv2.rectangle(frame, (10, 100), (450, 260), (0,0,0), -1)
            cv2.putText(frame, f"PLATE: {ultima_placa_validada}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            
            if datos_propietario_actual.get("estado") == "REGISTRADO":
                cv2.putText(frame, f"Owner: {datos_propietario_actual['nombre']}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.putText(frame, f"Phone: {datos_propietario_actual['celular']}", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
                cv2.rectangle(frame, (10, 100), (450, 260), (0,255,0), 2) 
            else:
                cv2.putText(frame, "NOT REGISTERED", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.rectangle(frame, (10, 100), (450, 260), (0,0,255), 2)

        # -----------------------
        # DETONANTE
        # -----------------------
        if accion_tirar_detectada:
            print("🚨 ACTION DETECTED -> STARTING CAPTURE")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if coords_placa_actual:
                px1, py1, px2, py2 = coords_placa_actual
                roi_placa = frame[py1:py2, px1:px2]
                if roi_placa.size > 0:
                    nombre_archivo_placa = f"evidencias/placas/PLATE_{ultima_placa_validada}_{timestamp}.jpg"
                    cv2.imwrite(nombre_archivo_placa, roi_placa)
                    print(f"📸 Plate captured: {nombre_archivo_placa}")
            
            cv2.imwrite(f"evidencias/evidencia_completa_{timestamp}.jpg", frame)

            fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
            out = cv2.VideoWriter(f"evidencias/videos/video_{timestamp}.avi", fourcc, 20.0, (frame_width, frame_height))
            
            GRABANDO = True
            inicio_grabacion = current_time

    # FPS
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Sistema Policial", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
if out: out.release()
cv2.destroyAllWindows()