# gui.py

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
        # Ejemplo de rangos para codo
        if lado == "izq":
            return 150, 170   # brazo izquierdo
        elif lado == "der":
            return 160, 180   # brazo derecho
        else:
            return 0, 180
    elif ejercicio == 2:
        # Ejemplo de rangos para rodilla
        if lado == "izq":
            return 160, 180   # rodilla izquierda
        elif lado == "der":
            return 170, 180   # rodilla derecha
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

        # Contador de repeticiones y estado (solo para brazo)
        self.reps = 0
        self.stage = "out"  # "in" cuando ángulo está dentro del rango, "out" en otro caso

        # --------------------------------------------------------
        # Configuración de la ventana principal
        # --------------------------------------------------------
        root.title("App de ejercicios con cámara")
        root.geometry("950x650")

        # Menú superior
        menubar = tk.Menu(root)
        root.config(menu=menubar)
        ayuda_menu = tk.Menu(menubar, tearoff=0)
        ayuda_menu.add_command(label="Acerca de", command=self.mostrar_acerca)
        menubar.add_cascade(label="Ayuda", menu=ayuda_menu)

        # --------------------------------------------------------
        # Panel lateral izquierdo: controles
        # --------------------------------------------------------
        frame_left = ttk.Frame(root, padding=10)
        frame_left.grid(row=0, column=0, sticky="ns")

        # 1) Selección de ejercicio (brazo o pierna)
        ttk.Label(frame_left, text="Seleccione el ejercicio").pack(pady=(5, 2))
        ttk.Radiobutton(
            frame_left, text="Parte Superior (Brazos)",
            variable=self.ejercicio_var, value=1,
            command=self.reset_counters
        ).pack(anchor="w", pady=2)
        ttk.Radiobutton(
            frame_left, text="Parte Inferior (Piernas)",
            variable=self.ejercicio_var, value=2,
            command=self.reset_counters
        ).pack(anchor="w", pady=2)

        # 2) Selección de lado (izquierdo / derecho)
        ttk.Label(frame_left, text="Seleccione el lado").pack(pady=(20, 2))
        self.side_var = tk.StringVar(value="izq")
        ttk.Radiobutton(
            frame_left, text="Izquierdo",
            variable=self.side_var, value="izq",
            command=self.reset_counters
        ).pack(anchor="w", pady=2)
        ttk.Radiobutton(
            frame_left, text="Derecho",
            variable=self.side_var, value="der",
            command=self.reset_counters
        ).pack(anchor="w", pady=2)

        # 3) Botones de control
        ttk.Button(frame_left, text="Actualizar cámaras", command=self.actualizar_camaras).pack(pady=(30, 5))
        ttk.Button(frame_left, text="Iniciar", command=self.iniciar_captura).pack(pady=5)
        ttk.Button(frame_left, text="Salir", command=self.cerrar).pack(pady=(30, 0))

        # --------------------------------------------------------
        # Panel central: lienzo donde irá el video
        # --------------------------------------------------------
        self.panel = tk.Canvas(root, bg="black", highlightthickness=0)
        self.panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)

        # Label para feedback de texto debajo del Canvas
        self.feedback_label = ttk.Label(root, text="", font=("Arial", 14))
        self.feedback_label.grid(row=1, column=1, pady=(0, 5))

        # Label para contador de repeticiones (solo para brazo)
        self.counter_label = ttk.Label(root, text="Reps: 0", font=("Arial", 14))
        self.counter_label.grid(row=2, column=1, pady=(0, 10))

        # --------------------------------------------------------
        # ComboBox para elegir cámara
        # --------------------------------------------------------
        self.camaras = listar_camaras()
        cam_text = str(self.camaras[0]) if self.camaras else "0"
        self.camara_var = tk.StringVar(value=cam_text)
        ttk.Label(frame_left, text="Fuente de video:").pack(pady=(10, 2))
        self.combo_camaras = ttk.Combobox(
            frame_left, values=[str(c) for c in self.camaras],
            textvariable=self.camara_var, state="readonly", width=8
        )
        self.combo_camaras.pack(pady=2)

        if not self.camaras:
            messagebox.showwarning("Advertencia", "No se detectaron cámaras disponibles.")

        # Vincula el evento de cambio de tamaño para escalar la imagen
        self.panel.bind("<Configure>", self.on_resize)

    def mostrar_acerca(self):
        messagebox.showinfo(
            "Acerca de",
            "App de ejercicios con cámara.\nDesarrollado por CodeLab Freelancers."
        )

    def reset_counters(self):
        """
        Reinicia el contador de reps y estado cuando cambies ejercicio o lado.
        """
        self.reps = 0
        self.stage = "out"
        self.counter_label.config(text="Reps: 0")

    def on_resize(self, event):
        """
        Cuando el Canvas cambia de tamaño, guarda las nuevas dimensiones
        y vuelve a dibujar el último frame.
        """
        self.panel_width = max(100, event.width)
        self.panel_height = max(100, event.height)
        self.mostrar_frame_actual()

    def mostrar_frame_actual(self):
        """
        Toma el último frame procesado y lo redimensiona para que encaje
        en el Canvas, manteniendo proporciones.
        """
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
        self.panel.imgtk = imgtk  # para que no lo recoja el GC

        if not hasattr(self, 'canvas_image_id'):
            self.canvas_image_id = self.panel.create_image(
                panel_w // 2, panel_h // 2, image=imgtk
            )
        else:
            self.panel.itemconfig(self.canvas_image_id, image=imgtk)
            self.panel.coords(self.canvas_image_id, panel_w // 2, panel_h // 2)

    def update_frame(self):
        """
        Captura un frame de la cámara, lo procesa con MediaPipe,
        obtiene el ángulo según ejercicio y lado seleccionados,
        cuenta una repetición cada vez que el ángulo entra en el rango y luego sale,
        y muestra feedback.
        """
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self._after_id = self.root.after(100, self.update_frame)
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = pose.process(frame_rgb)

        angulo_detectado = None

        if resultados.pose_landmarks:
            # Dibujar todos los landmarks
            mp_drawing.draw_landmarks(
                frame, resultados.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

            lado = self.side_var.get()
            ejercicio = self.ejercicio_var.get()

            if ejercicio == 1:
                # Parte superior: detectar codo del lado elegido
                angulo_detectado = detectar_codo(
                    resultados.pose_landmarks.landmark, lado
                )
            else:
                # Parte inferior: detectar rodilla del lado elegido
                angulo_detectado = detectar_rodilla(
                    resultados.pose_landmarks.landmark, lado
                )

        # Obtener rango según ejercicio y lado
        ang_min, ang_max = obtener_rango_ejercicio(
            self.ejercicio_var.get(), self.side_var.get()
        )

        # Depuración en consola (opcional)
        print(f"Ejercicio={self.ejercicio_var.get()}, Lado={self.side_var.get()}, Ángulo={angulo_detectado}")

        # -----------------------------------------------------------
        # Contador de repeticiones (solo para ejercicio==1, brazo),
        # contando cada vez que el ángulo entra en el rango y sale.
        # -----------------------------------------------------------
        if self.ejercicio_var.get() == 1:
            if angulo_detectado is not None:
                # Si antes estábamos fuera y ahora entra en rango, actualizamos estado a "in"
                if self.stage == "out" and ang_min <= angulo_detectado <= ang_max:
                    self.stage = "in"
                # Si antes estábamos adentro y ahora sale del rango, contamos 1 rep y volvemos a "out"
                elif self.stage == "in" and not (ang_min <= angulo_detectado <= ang_max):
                    self.reps += 1
                    self.stage = "out"
                    self.counter_label.config(text=f"Reps: {self.reps}")

        # Generar texto de feedback
        feedback_text, feedback_color = feedback_ejercicio(
            angulo_detectado, ang_min, ang_max
        )

        # Guardar y mostrar frame
        self.last_frame = frame.copy()
        self.mostrar_frame_actual()

        # Mostrar feedback textual
        self.feedback_label.config(
            text=feedback_text,
            foreground=self.rgb_to_hex(feedback_color)
        )

        # Programar siguiente frame
        self._after_id = self.root.after(10, self.update_frame)

    def rgb_to_hex(self, rgb):
        """
        Convierte una tupla BGR o RGB a string "#rrggbb" para Tkinter.
        """
        return "#{:02x}{:02x}{:02x}".format(rgb[2], rgb[1], rgb[0])

    def cerrar(self):
        """
        Detiene la captura, libera la cámara y cierra la ventana.
        """
        self.running = False
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

    def actualizar_camaras(self):
        """
        Refresca la lista de cámaras disponibles y actualiza el Combobox.
        """
        self.camaras = listar_camaras()
        self.combo_camaras['values'] = [str(c) for c in self.camaras]
        if self.camaras:
            self.camara_var.set(str(self.camaras[0]))

    def iniciar_captura(self):
        """
        Intenta abrir la cámara seleccionada y comienza el loop de update_frame().
        """
        if not self.camaras:
            messagebox.showerror("Error", "No hay cámaras disponibles.")
            return

        # Si ya estaba corriendo, lo detenemos primero
        self.running = False
        if hasattr(self, '_after_id') and self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except Exception:
                pass

        # Liberar cámara previa
        if self.cap is not None and self.cap.isOpened():
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

        # Reiniciamos contador y estado
        self.reset_counters()

        self.running = True
        self.update_frame()
