import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import mediapipe as mp

from pose_utils import detectar_codo, detectar_rodilla


def listar_camaras(max_camaras=5):
    """
    Detecta índices de cámaras disponibles (0, 1, 2, ...) hasta max_camaras.
    """
    disponibles = []
    for i in range(max_camaras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            disponibles.append(i)
            cap.release()
    return disponibles


def feedback_ejercicio(angulo, angulo_min, angulo_max):
    """
    Dado un ángulo detectado y un rango [angulo_min, angulo_max],
    devuelve un texto y un color (BGR) según si el ángulo está dentro del rango.
    """
    if angulo is None:
        return "Articulación no visible", (200, 200, 0)  # amarillo suave

    if angulo_min <= angulo <= angulo_max:
        return "Ejercicio correcto", (0, 255, 0)
    else:
        return f"Ángulo fuera de rango [{angulo_min}, {angulo_max}]", (0, 0, 255)


def obtener_rango_ejercicio(ejercicio, lado):
    """
    Devuelve (angulo_min, angulo_max) según:
      - ejercicio == 1: parte superior (codo)
          · lado == "izq": rango para brazo izquierdo
          · lado == "der": rango para brazo derecho
      - ejercicio == 2: parte inferior (rodilla)
          · lado == "izq": rango para rodilla izquierda
          · lado == "der": rango para rodilla derecha
    Ajusta estos valores al rango real que necesites.
    """
    if ejercicio == 1:
        if lado == "izq":
            return 150, 170
        elif lado == "der":
            return 160, 180
        else:
            return 0, 180
    elif ejercicio == 2:
        if lado == "izq":
            return 160, 180
        elif lado == "der":
            return 170, 180
        else:
            return 0, 180
    else:
        return 0, 360


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


class PoseAppGUI:
    def __init__(self, root, on_exit, ejercicio_var):
        self.root = root
        self.ejercicio_var = ejercicio_var  # 1=brazo, 2=pierna
        self.cap = None
        self.running = False
        self.last_frame = None
        self.current_cam_index = None

        # Variables para modelo de series y repeticiones
        self.target_reps_var = tk.IntVar(value=10)
        self.target_series_var = tk.IntVar(value=3)

        # Contador de repeticiones y estado
        self.reps = 0
        self.stage = "out"
        self.current_target_reps = self.target_reps_var.get()
        self.series_left = self.target_series_var.get()

        # --------------------------------------------------------
        root.title("App de ejercicios con cámara")
        root.geometry("1000x700")

        menubar = tk.Menu(root)
        root.config(menu=menubar)
        ayuda_menu = tk.Menu(menubar, tearoff=0)
        ayuda_menu.add_command(label="Acerca de", command=self.mostrar_acerca)
        menubar.add_cascade(label="Ayuda", menu=ayuda_menu)

        frame_left = ttk.Frame(root, padding=10)
        frame_left.grid(row=0, column=0, sticky="ns")

        ttk.Label(frame_left, text="Seleccione el ejercicio").pack(pady=(5, 2))
        ttk.Radiobutton(
            frame_left,
            text="Parte Superior (Brazos)",
            variable=self.ejercicio_var,
            value=1,
            command=self.reset_counters
        ).pack(anchor="w", pady=2)
        ttk.Radiobutton(
            frame_left,
            text="Parte Inferior (Piernas)",
            variable=self.ejercicio_var,
            value=2,
            command=self.reset_counters
        ).pack(anchor="w", pady=2)

        ttk.Label(frame_left, text="Seleccione el lado").pack(pady=(20, 2))
        self.side_var = tk.StringVar(value="izq")
        ttk.Radiobutton(
            frame_left,
            text="Izquierdo",
            variable=self.side_var,
            value="izq",
            command=self.reset_counters
        ).pack(anchor="w", pady=2)
        ttk.Radiobutton(
            frame_left,
            text="Derecho",
            variable=self.side_var,
            value="der",
            command=self.reset_counters
        ).pack(anchor="w", pady=2)

        ttk.Label(frame_left, text="Reps por serie:").pack(pady=(20, 2))
        ttk.Entry(frame_left, textvariable=self.target_reps_var, width=10).pack(pady=2)
        ttk.Label(frame_left, text="Series:").pack(pady=(10, 2))
        ttk.Entry(frame_left, textvariable=self.target_series_var, width=10).pack(pady=2)

        ttk.Button(frame_left, text="Actualizar cámaras", command=self.actualizar_camaras).pack(pady=(30, 5))
        ttk.Button(frame_left, text="Iniciar", command=self.iniciar_captura).pack(pady=5)
        ttk.Button(frame_left, text="Salir", command=self.cerrar).pack(pady=(30, 0))

        self.panel = tk.Canvas(root, bg="black", highlightthickness=0)
        self.panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)

        self.feedback_label = ttk.Label(root, text="", font=("Arial", 14))
        self.feedback_label.grid(row=1, column=1, pady=(0, 5))

        self.counter_label = ttk.Label(root, text="Reps: 0", font=("Arial", 14))
        self.counter_label.grid(row=2, column=1, pady=(0, 5))
        self.series_label = ttk.Label(
            root,
            text=f"Series restantes: {self.series_left}",
            font=("Arial", 14)
        )
        self.series_label.grid(row=3, column=1, pady=(0, 10))

        self.camaras = listar_camaras()
        cam_text = str(self.camaras[0]) if self.camaras else "0"
        self.camara_var = tk.StringVar(value=cam_text)
        ttk.Label(frame_left, text="Fuente de video:").pack(pady=(10, 2))
        self.combo_camaras = ttk.Combobox(
            frame_left,
            values=[str(c) for c in self.camaras],
            textvariable=self.camara_var,
            state="readonly",
            width=8
        )
        self.combo_camaras.pack(pady=2)

        if not self.camaras:
            messagebox.showwarning("Advertencia", "No se detectaron cámaras disponibles.")

        self.panel.bind("<Configure>", self.on_resize)

    def mostrar_acerca(self):
        messagebox.showinfo(
            "Acerca de",
            "App de ejercicios con cámara.\nDesarrollado por CodeLab Freelancers."
        )

    def reset_counters(self):
        self.reps = 0
        self.stage = "out"
        self.current_target_reps = self.target_reps_var.get()
        self.series_left = self.target_series_var.get()
        self.counter_label.config(text="Reps: 0")
        self.series_label.config(text=f"Series restantes: {self.series_left}")

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

        panel_w = self.panel.winfo_width()
        panel_h = self.panel.winfo_height()
        orig_w, orig_h = img.size
        ratio = min(panel_w / orig_w, panel_h / orig_h)
        new_w = int(orig_w * ratio)
        new_h = int(orig_h * ratio)
        img = img.resize((new_w, new_h), Image.BILINEAR)

        imgtk = ImageTk.PhotoImage(image=img)
        self.panel.imgtk = imgtk

        if not hasattr(self, 'canvas_image_id'):
            self.canvas_image_id = self.panel.create_image(panel_w // 2, panel_h // 2, image=imgtk)
        else:
            self.panel.itemconfig(self.canvas_image_id, image=imgtk)
            self.panel.coords(self.canvas_image_id, panel_w // 2, panel_h // 2)

    def update_frame(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self._after_id = self.root.after(100, self.update_frame)
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = pose.process(frame_rgb)

        angulo_detectado = None

        if resultados.pose_landmarks:
            mp_drawing.draw_landmarks(frame, resultados.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lado = self.side_var.get()
            ejercicio = self.ejercicio_var.get()

            if ejercicio == 1:
                angulo_detectado = detectar_codo(resultados.pose_landmarks.landmark, lado)
            else:
                angulo_detectado = detectar_rodilla(resultados.pose_landmarks.landmark, lado)

        ang_min, ang_max = obtener_rango_ejercicio(
            self.ejercicio_var.get(),
            self.side_var.get()
        )

        print(f"Ejercicio={self.ejercicio_var.get()}, Lado={self.side_var.get()}, Ángulo={angulo_detectado}")

        # ...existing code...
        if self.running and self.ejercicio_var.get() == 1:
            if angulo_detectado is not None:
                if self.stage == "out" and ang_min <= angulo_detectado <= ang_max:
                    self.stage = "in"
                elif self.stage == "in" and not (ang_min <= angulo_detectado <= ang_max):
                    self.reps += 1
                    self.stage = "out"
                    if self.reps >= self.current_target_reps:
                        self.series_left -= 1
                        self.series_label.config(text=f"Series restantes: {self.series_left}")
                        if self.series_left <= 0:
                            self.running = False
                            self.feedback_label.config(text="¡Ejercicio completado!", foreground="#008000")
                            self.counter_label.config(text=f"Reps: {self.current_target_reps}")
                            messagebox.showinfo("Completado", "¡Has completado todas las series del ejercicio!")
                        else:
                            self.reps = 0
                            self.counter_label.config(text=f"Reps: {self.reps}")
                    else:
                        self.counter_label.config(text=f"Reps: {self.reps}")
        # ...existing code...

        # Guardar y mostrar el último frame siempre
        self.last_frame = frame.copy()
        self.mostrar_frame_actual()

        feedback_text, feedback_color = feedback_ejercicio(angulo_detectado, ang_min, ang_max)
        self.feedback_label.config(text=feedback_text, foreground=self.rgb_to_hex(feedback_color))

        # Programar siguiente frame independientemente de `running`
        self._after_id = self.root.after(10, self.update_frame)

    def rgb_to_hex(self, rgb):
        return "#{:02x}{:02x}{:02x}".format(rgb[2], rgb[1], rgb[0])

    def cerrar(self):
        self.running = False
        if self.cap is not None and self.cap.isOpened():
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

        # Detener conteo previo si estaba activo
        self.running = False
        if hasattr(self, '_after_id') and self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except Exception:
                pass

        # Liberar cámara anterior
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None

        try:
            cam_index = int(self.camara_var.get())
        except ValueError:
            messagebox.showerror("Error", "Fuente de cámara inválida.")
            return

        self.current_cam_index = cam_index
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"No se pudo abrir la cámara {cam_index}.")
            self.cap = None
            return

        # Reiniciar contadores y etiquetas
        self.reset_counters()
        self.feedback_label.config(text="")
        self.counter_label.config(text="Reps: 0")

        # Empezar el conteo y la captura de fotogramas
        self.running = True
        self.update_frame()
