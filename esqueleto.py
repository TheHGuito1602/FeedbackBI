import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def calcular_angulo(p1, p2, p3):
    angulo = np.degrees(np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - np.arctan2(p1[1] - p2[1], p1[0] - p2[0]))
    if angulo < 0:
        angulo += 360
    return angulo

def feedback_ejercicio(angulo):
    if 160 <= angulo <= 180:
        return "Ejercicio correcto", (0, 255, 0)
    else:
        return "Corrige el ángulo. Debe estar recto.", (0, 0, 255)

def articulacion_visible(landmarks, articulacion):
    return landmarks[articulacion].visibility > 0.5

def detectar_parte_superior(landmarks, frame):
    hombro_izq = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
    codo_izq = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
    muñeca_izq = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]

    hombro_der = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
    codo_der = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
    muñeca_der = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]

    brazo_izq_visible = articulacion_visible(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER) and \
                        articulacion_visible(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW) and \
                        articulacion_visible(landmarks, mp_pose.PoseLandmark.LEFT_WRIST)

    brazo_der_visible = articulacion_visible(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER) and \
                        articulacion_visible(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW) and \
                        articulacion_visible(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST)

    if brazo_izq_visible:
        angulo_codo_izq = calcular_angulo(hombro_izq, codo_izq, muñeca_izq)
        mensaje_izq, color_izq = feedback_ejercicio(angulo_codo_izq)
        cv2.putText(frame, f'B. IZQ. = {mensaje_izq}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_izq, 2, cv2.LINE_AA)

    if brazo_der_visible:
        angulo_codo_der = calcular_angulo(hombro_der, codo_der, muñeca_der)
        mensaje_der, color_der = feedback_ejercicio(angulo_codo_der)
        cv2.putText(frame, f'B. DER. = {mensaje_der}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color_der, 2, cv2.LINE_AA)

def detectar_parte_inferior(landmarks, frame):
    cadera_izq = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
    rodilla_izq = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
    tobillo_izq = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]

    cadera_der = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
    rodilla_der = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
    tobillo_der = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]

    pierna_izq_visible = articulacion_visible(landmarks, mp_pose.PoseLandmark.LEFT_HIP) and \
                          articulacion_visible(landmarks, mp_pose.PoseLandmark.LEFT_KNEE) and \
                          articulacion_visible(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)

    pierna_der_visible = articulacion_visible(landmarks, mp_pose.PoseLandmark.RIGHT_HIP) and \
                          articulacion_visible(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE) and \
                          articulacion_visible(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)

    if pierna_izq_visible:
        angulo_rodilla_izq = calcular_angulo(cadera_izq, rodilla_izq, tobillo_izq)
        mensaje_izq, color_izq = feedback_ejercicio(angulo_rodilla_izq)
        cv2.putText(frame, f'P. IZQ. = {mensaje_izq}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_izq, 2, cv2.LINE_AA)

    if pierna_der_visible:
        angulo_rodilla_der = calcular_angulo(cadera_der, rodilla_der, tobillo_der)
        mensaje_der, color_der = feedback_ejercicio(angulo_rodilla_der)
        cv2.putText(frame, f'P. DER. = {mensaje_der}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color_der, 2, cv2.LINE_AA)

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Ejercicios con cámara")
        self.ejercicio_var = tk.IntVar(value=1)
        self.cap = cv2.VideoCapture(0)
        self.running = True

        # Panel de selección
        frame_left = tk.Frame(root)
        frame_left.pack(side=tk.LEFT, padx=10, pady=10)
        tk.Label(frame_left, text="Seleccione el ejercicio").pack(pady=10)
        tk.Radiobutton(frame_left, text="Parte Superior (Brazos)", variable=self.ejercicio_var, value=1).pack(pady=5)
        tk.Radiobutton(frame_left, text="Parte Inferior (Piernas)", variable=self.ejercicio_var, value=2).pack(pady=5)
        tk.Button(frame_left, text="Salir", command=self.cerrar).pack(pady=10)

        # Panel de cámara
        self.panel = tk.Label(root)
        self.panel.pack(side=tk.LEFT, padx=10, pady=10)

        self.update_frame()

    def update_frame(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultados = pose.process(frame_rgb)
            if resultados.pose_landmarks:
                if self.ejercicio_var.get() == 1:
                    detectar_parte_superior(resultados.pose_landmarks.landmark, frame)
                elif self.ejercicio_var.get() == 2:
                    detectar_parte_inferior(resultados.pose_landmarks.landmark, frame)
                mp_drawing.draw_landmarks(frame, resultados.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)
        self.root.after(10, self.update_frame)

    def cerrar(self):
        self.running = False
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()