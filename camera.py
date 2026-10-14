"""
camera.py - abertura das cameras e captura em segundo plano.

abrir_camera() e a mesma logica que estava no Main.py (MJPG, buffer 1,
confere se o driver aceitou), so que agora serve pras duas cameras.

CameraThread existe por causa da SEGUNDA camera: cap.read() bloqueia ate o
proximo frame chegar (~33ms a 30fps). Se o laco principal lesse as duas em
sequencia, o FPS cairia pela metade e o PID ficaria mais lento. Com a thread,
a camera de cone fica lendo sozinha e o laco principal so pega o ultimo
frame pronto, sem esperar.
"""

import threading
import time
import cv2


def abrir_camera(indice, res, nome="camera", fps=30):
    cap = cv2.VideoCapture(indice, cv2.CAP_V4L2)

    # MJPG chega comprimido: ate 5-10x mais rapido que YUYV em USB.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # sempre o frame mais novo

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_real = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera {nome} (id {indice}): {w}x{h} @ {fps_real:.0f}fps pedido, "
          f"formato={fourcc_str}")
    if cap.isOpened() and fourcc_str.strip() != "MJPG":
        print(f"AVISO: camera {nome} NAO aceitou MJPG. Com duas cameras no "
              f"mesmo USB isso pode estourar a banda. Rode "
              f"'v4l2-ctl --list-formats-ext -d /dev/video{indice}'.")
    return cap


class CameraThread:
    """Le a camera em segundo plano e guarda so o frame mais recente."""

    def __init__(self, cap):
        self.cap = cap
        self._frame = None
        self._idx = 0
        self._t_frame = 0.0
        self._lock = threading.Lock()
        self._rodando = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._rodando:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
                self._idx += 1
                self._t_frame = time.time()

    def read(self):
        """Retorna (frame, numero_do_frame, idade_em_segundos).
        frame e None se ainda nao chegou nenhum."""
        with self._lock:
            if self._frame is None:
                return None, 0, float("inf")
            return self._frame, self._idx, time.time() - self._t_frame

    def release(self):
        self._rodando = False
        self._thread.join(timeout=1.0)
        self.cap.release()
