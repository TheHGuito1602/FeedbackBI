import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import mediapipe as mp

from pose_utils import detectar_parte_superior, detectar_parte_inferior

def listar_camaras(max_camaras=5):
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
        self.last_frame = None

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
        self.panel = tk.Canvas(root, bg="black", highlightthickness=0)
        self.panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # Label para feedback debajo del panel de cámara
        self.feedback_label = ttk.Label(root, text="", font=("Arial", 14))
        self.feedback_label.grid(row=1, column=1, pady=(0, 10))

        # Hacer la columna 1 expandible
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)

        # Vincula el evento de cambio de tamaño
        self.panel.bind("<Configure>", self.on_resize)

    def mostrar_acerca(self):
        messagebox.showinfo("Acerca de", "App de ejercicios con cámara.\nDesarrollado por CodeLab Freelancers.")

    def on_resize(self, event):
        self.panel_width = max(100, event.width)
        self.panel_height = max(100, event.height)
        self.mostrar_frame_actual()

    def mostrar_frame_actual(self):
        if self.last_frame is None:
            return

        frame = self.last_frame.copy()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        # Obtener el tamaño actual del canvas
        panel_w = self.panel.winfo_width()
        panel_h = self.panel.winfo_height()

        orig_w, orig_h = img.size
        ratio = min(panel_w / orig_w, panel_h / orig_h)
        new_w = int(orig_w * ratio)
        new_h = int(orig_h * ratio)
        img = img.resize((new_w, new_h), Image.BILINEAR)

        imgtk = ImageTk.PhotoImage(image=img)
        self.panel.imgtk = imgtk  # evita el garbage collection

        if not hasattr(self, 'canvas_image_id'):
            self.canvas_image_id = self.panel.create_image(panel_w // 2, panel_h // 2, image=imgtk)
        else:
            self.panel.itemconfig(self.canvas_image_id, image=imgtk)
            self.panel.coords(self.canvas_image_id, panel_w // 2, panel_h // 2)



    def update_frame(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self._after_id = self.root.after(100, self.update_frame)
            return

        # Procesamiento con MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = pose.process(frame_rgb)
        angulo = None
        feedback_text = ""
        feedback_color = (0, 0, 0)

        if resultados.pose_landmarks:
            if self.ejercicio_var.get() == 1:
                angulo = detectar_parte_superior(resultados.pose_landmarks.landmark, frame)
            elif self.ejercicio_var.get() == 2:
                angulo = detectar_parte_inferior(resultados.pose_landmarks.landmark, frame)
            mp_drawing.draw_landmarks(frame, resultados.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if angulo is not None:
            feedback_text, feedback_color = feedback_ejercicio(angulo)

        # Guardar el frame procesado
        self.last_frame = frame.copy()

        # Mostrar inmediatamente
        self.mostrar_frame_actual()

        # Mostrar feedback
        self.feedback_label.config(text=feedback_text, foreground=self.rgb_to_hex(feedback_color))

        # Programar el siguiente frame
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

        self.running = False
        if hasattr(self, '_after_id') and self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except Exception:
                pass

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
