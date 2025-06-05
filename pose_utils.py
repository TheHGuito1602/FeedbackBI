# pose_utils.py

import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

def calcular_angulo(p1, p2, p3):
    """
    Calcula el ángulo interior formado por los puntos p1-p2-p3 (p2 es el vértice).
    Devuelve siempre un valor ≤ 180°.
    """
    ang = np.degrees(
        np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) -
        np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    )
    if ang < 0:
        ang += 360
    if ang > 180:
        ang = 360 - ang
    return ang

def articulacion_visible(landmarks, articulacion):
    """
    Verifica si una articulación (landmark) está suficientemente visible.
    Umbral reducido a 0.3 para captar mejor ambos lados.
    """
    return landmarks[articulacion].visibility > 0.3

def detectar_codo(landmarks, lado):
    """
    Dado el parámetro 'lado' ("izq" o "der"), intenta detectar hombro, codo y muñeca
    de ese lado y devuelve el ángulo interior del codo.
    Si no alcanza visibilidad, devuelve None.
    """
    if lado == "izq":
        h = mp_pose.PoseLandmark.LEFT_SHOULDER
        e = mp_pose.PoseLandmark.LEFT_ELBOW
        w = mp_pose.PoseLandmark.LEFT_WRIST
    else:  # lado == "der"
        h = mp_pose.PoseLandmark.RIGHT_SHOULDER
        e = mp_pose.PoseLandmark.RIGHT_ELBOW
        w = mp_pose.PoseLandmark.RIGHT_WRIST

    if (
        articulacion_visible(landmarks, h)
        and articulacion_visible(landmarks, e)
        and articulacion_visible(landmarks, w)
    ):
        hombro = [landmarks[h].x, landmarks[h].y]
        codo    = [landmarks[e].x, landmarks[e].y]
        muñeca  = [landmarks[w].x, landmarks[w].y]
        return calcular_angulo(hombro, codo, muñeca)
    else:
        return None

def detectar_rodilla(landmarks, lado):
    """
    Dado 'lado' ("izq" o "der"), intenta detectar cadera, rodilla y tobillo
    de ese lado y devuelve el ángulo interior de la rodilla.
    Si no alcanza visibilidad, devuelve None.
    """
    if lado == "izq":
        h = mp_pose.PoseLandmark.LEFT_HIP
        k = mp_pose.PoseLandmark.LEFT_KNEE
        a = mp_pose.PoseLandmark.LEFT_ANKLE
    else:  # lado == "der"
        h = mp_pose.PoseLandmark.RIGHT_HIP
        k = mp_pose.PoseLandmark.RIGHT_KNEE
        a = mp_pose.PoseLandmark.RIGHT_ANKLE

    if (
        articulacion_visible(landmarks, h)
        and articulacion_visible(landmarks, k)
        and articulacion_visible(landmarks, a)
    ):
        cadera  = [landmarks[h].x, landmarks[h].y]
        rodilla = [landmarks[k].x, landmarks[k].y]
        tobillo = [landmarks[a].x, landmarks[a].y]
        return calcular_angulo(cadera, rodilla, tobillo)
    else:
        return None
