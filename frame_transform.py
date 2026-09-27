# frame_transform.py
import cv2

def prepare_frame_for_model(frame_bgr, altitude, ref_altitude, input_size):
    """Приводит кадр к виду 'как будто снят с высоты ref_altitude'.
    input_size — сторона квадратного входа энкодера (P)."""
    P = input_size
    scale = altitude / ref_altitude            # доля входа, занимаемая кадром
    new_size = max(1, int(round(P * scale)))

    img = cv2.resize(frame_bgr, (new_size, new_size), interpolation=cv2.INTER_AREA)

    if new_size < P:                           # дрон НИЖЕ эталона
        pad = (P - new_size) // 2
        img = cv2.copyMakeBorder(img, pad, P - new_size - pad,
                                 pad, P - new_size - pad,
                                 cv2.BORDER_CONSTANT, value=0)
    elif new_size > P:                         # дрон ВЫШЕ эталона
        off = (new_size - P) // 2
        img = img[off:off + P, off:off + P]
    return img                                 # всегда ровно P x P
