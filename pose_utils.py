import numpy as np
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

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

    # Retorna el ángulo del brazo izquierdo si está visible, si no el derecho, si no None
    if brazo_izq_visible:
        angulo_codo_izq = calcular_angulo(hombro_izq, codo_izq, muñeca_izq)
        return angulo_codo_izq
    elif brazo_der_visible:
        angulo_codo_der = calcular_angulo(hombro_der, codo_der, muñeca_der)
        return angulo_codo_der
    else:
        return None

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

    # Retorna el ángulo de la pierna izquierda si está visible, si no el derecho, si no None
    if pierna_izq_visible:
        angulo_rodilla_izq = calcular_angulo(cadera_izq, rodilla_izq, tobillo_izq)
        return angulo_rodilla_izq
    elif pierna_der_visible:
        angulo_rodilla_der = calcular_angulo(cadera_der, rodilla_der, tobillo_der)
        return angulo_rodilla_der
    else:
        return None