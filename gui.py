import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import mediapipe as mp

from pose_utils import detectar_parte_superior, detectar_parte_inferior

def listar_camaras(max_camaras=5):
    import cv2
    disponibles = []
    for i in range(max_camaras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            disponibles.append(i)
            cap.release()
    return disponibles

def feedback_ejercicio(angulo):
    if 160 <= angulo <= 180:
        return "Ejercicio correcto", (0, 255, 0)
    else:
        return "Corrige el ángulo. Debe estar recto.", (255, 0, 0)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

class PoseAppGUI:
    def __init__(self, root, on_exit, ejercicio_var):
        self.root = root
        self.ejercicio_var = ejercicio_var
        self.cap = None
        self.running = False

        # Menú superior
        menubar = tk.Menu(root)
        root.config(menu=menubar)
        ayuda_menu = tk.Menu(menubar, tearoff=0)
        ayuda_menu.add_command(label="Acerca de", command=self.mostrar_acerca)
        menubar.add_cascade(label="Ayuda", menu=ayuda_menu)

        # Panel lateral
        frame_left = ttk.Frame(root, padding=10)
        frame_left.grid(row=0, column=0, sticky="ns")
        ttk.Label(frame_left, text="Seleccione el ejercicio").pack(pady=10)
        ttk.Radiobutton(frame_left, text="Parte Superior (Brazos)", variable=self.ejercicio_var, value=1).pack(pady=5)
        ttk.Radiobutton(frame_left, text="Parte Inferior (Piernas)", variable=self.ejercicio_var, value=2).pack(pady=5)

        # Selección de cámara
        self.camaras = listar_camaras()
        self.camara_var = tk.StringVar(value=str(self.camaras[0]) if self.camaras else "0")
        if not self.camaras:
            messagebox.showwarning("Advertencia", "No se detectaron cámaras disponibles.")
        ttk.Label(frame_left, text="Fuente de video:").pack(pady=5)
        self.combo_camaras = ttk.Combobox(frame_left, values=self.camaras, textvariable=self.camara_var, state="readonly")
        self.combo_camaras.pack(pady=5)
        ttk.Button(frame_left, text="Actualizar cámaras", command=self.actualizar_camaras).pack(pady=5)
        ttk.Button(frame_left, text="Iniciar", command=self.iniciar_captura).pack(pady=5)
        ttk.Button(frame_left, text="Salir", command=self.cerrar).pack(pady=10)

        # Panel de cámara
        self.panel = ttk.Label(root)
        self.panel.grid(row=0, column=1, padx=10, pady=10)

        # Label para feedback debajo del panel de cámara
        self.feedback_label = ttk.Label(root, text="", font=("Arial", 14))
        self.feedback_label.grid(row=1, column=1, pady=(0, 10))

    def mostrar_acerca(self):
        messagebox.showinfo("Acerca de", "App de ejercicios con cámara.\nDesarrollado por Hugo.")

    def update_frame(self):
        if not self.running or self.cap is None:
            return
        ret, frame = self.cap.read()
        feedback_text = ""
        feedback_color = (0, 0, 0)
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultados = pose.process(frame_rgb)
            angulo = None
            if resultados.pose_landmarks:
                if self.ejercicio_var.get() == 1:
                    angulo = detectar_parte_superior(resultados.pose_landmarks.landmark, frame)
                elif self.ejercicio_var.get() == 2:
                    angulo = detectar_parte_inferior(resultados.pose_landmarks.landmark, frame)
                mp_drawing.draw_landmarks(frame, resultados.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            if angulo is not None:
                feedback_text, feedback_color = feedback_ejercicio(angulo)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((1280, 800))  # <-- AQUÍ defines el tamaño de la cámara
            imgtk = ImageTk.PhotoImage(image=img)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)
        # Actualiza el label de feedback
        self.feedback_label.config(text=feedback_text, foreground=self.rgb_to_hex(feedback_color))
        self._after_id = self.root.after(10, self.update_frame)

    def rgb_to_hex(self, rgb):
        return "#%02x%02x%02x" % rgb

    def cerrar(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()
    
    def actualizar_camaras(self):
        self.camaras = listar_camaras()
        self.combo_camaras['values'] = [str(c) for c in self.camaras]
        if self.camaras:
            self.camara_var.set(str(self.camaras[0]))
    
    def iniciar_captura(self):
        if not self.camaras:
            messagebox.showerror("Error", "No hay cámaras disponibles.")
            return

        # Detener captura anterior si está corriendo
        self.running = False
        if hasattr(self, '_after_id') and self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except Exception:
                pass  # Si no es válido, simplemente ignora

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        try:
            cam_index = int(self.camara_var.get())
        except ValueError:
            messagebox.showerror("Error", "Fuente de cámara inválida.")
            return

        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"No se pudo abrir la cámara {cam_index}.")
            self.cap = None
            return

        self.running = True
        self.update_frame()