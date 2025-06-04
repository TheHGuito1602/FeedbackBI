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
        #ttk.Button(frame_left, text="Salir", command=self.cerrar).pack(pady=10)

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

        #self.update_frame()

    def mostrar_acerca(self):
        messagebox.showinfo("Acerca de", "App de ejercicios con cámara.\nDesarrollado por Hugo.")

    def update_frame(self):
        if not self.running or self.cap is None:
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
        if self.cap is not None:
            self.cap.release()
        try:
            cam_index = int(self.camara_var.get())
        except ValueError:
            messagebox.showerror("Error", "Fuente de cámara inválida.")
            return
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"No se pudo abrir la cámara {cam_index}.")
            self.cap = None
            self.running = False
            return
        self.running = True
        self.update_frame()