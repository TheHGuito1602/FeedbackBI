import tkinter as tk
from gui import PoseAppGUI

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1480x900")  # <-- Tamaño inicial (ancho x alto)
    ejercicio_var = tk.IntVar(value=1)
    app = PoseAppGUI(root, root.destroy, ejercicio_var)
    root.mainloop()