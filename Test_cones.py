"""
test_cones.py - testa SO a deteccao de cone. Nao toca em GPIO, nao liga motor.
Use pra calibrar a cor do cone e as distancias LONGE/PERTO com o robo na mao.

    python3 Test_cones.py              # camera de cone (config.CAMERA_CONE_ID)
    python3 Test_cones.py 2            # camera de indice 2
    python3 Test_cones.py video.mp4    # gravacao
    python3 Test_cones.py foto.jpg     # imagem parada

Como calibrar:
  1. Cone no chao, na distancia onde o carro deve COMECAR a desviar.
     Anote "altura" -> CONE_ALTURA_LONGE
  2. Cone na distancia onde o desvio ja deve ser total.
     Anote "altura" -> CONE_ALTURA_PERTO
  3. Se o cone nao aparece (ou aparece lixo): ajuste S/V com as teclas.

Teclas: 'q' sai | 'a'/'d' saturacao minima -/+ | 'w'/'s' brilho minimo +/-
"""

import sys
import cv2

import config
from cones import ConeVision


def hud(img, res):
    if res.found:
        txt = (f"CONE x {res.cx:+.2f} lat {res.lat:+.1f}w  altura {res.altura:.2f}  "
               f"prox {res.proximidade:.2f}")
        cor = (0, 255, 0)
    else:
        txt = f"sem cone  cand {res.n_candidatos}"
        cor = (0, 0, 255)
    cv2.putText(img, txt, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cor, 1)
    cv2.putText(img, f"S>={config.CONE_S_MIN}  V>={config.CONE_V_MIN}",
                (6, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 0), 1)
    return img


def tecla(k):
    if k == ord('d'):
        config.CONE_S_MIN = min(255, config.CONE_S_MIN + 10)
    elif k == ord('a'):
        config.CONE_S_MIN = max(0, config.CONE_S_MIN - 10)
    elif k == ord('w'):
        config.CONE_V_MIN = min(255, config.CONE_V_MIN + 10)
    elif k == ord('s'):
        config.CONE_V_MIN = max(0, config.CONE_V_MIN - 10)
    elif k == ord('q'):
        return False
    return True


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else str(config.CAMERA_CONE_ID)
    vision = ConeVision()

    # ---- imagem parada
    if arg.lower().endswith(('.jpg', '.jpeg', '.png')):
        frame = cv2.imread(arg)
        if frame is None:
            print("Nao consegui abrir a imagem.")
            return
        while True:
            res = vision.process(frame, debug=True)
            cv2.imshow("Cone", hud(res.debug_view, res))
            cv2.imshow("Mascara", res.debug_mask)
            if not tecla(cv2.waitKey(0) & 0xFF):
                break
        cv2.destroyAllWindows()
        return

    # ---- camera (indice numerico) ou video
    if arg.isdigit():
        cap = cv2.VideoCapture(int(arg), cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_CONE_RES[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_CONE_RES[1])
    else:
        cap = cv2.VideoCapture(arg)
    if not cap.isOpened():
        print("Camera/arquivo nao abriu.")
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = vision.process(frame, debug=True)
        cv2.imshow("Cone", hud(res.debug_view, res))
        cv2.imshow("Mascara", res.debug_mask)
        if not tecla(cv2.waitKey(1) & 0xFF):
            break

    print(f"\nCole no config.py:\n  CONE_S_MIN = {config.CONE_S_MIN}"
          f"\n  CONE_V_MIN = {config.CONE_V_MIN}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
