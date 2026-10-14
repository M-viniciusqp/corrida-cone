"""
test_vision.py - testa SO a visao. Nao toca em GPIO, nao liga motor.
Use isso pra calibrar SCAN_TOP_PCT e o branco com o robo parado na mao.

    python3 test_vision.py             # camera ao vivo
    python3 test_vision.py video.mp4   # gravacao
    python3 test_vision.py foto.jpg    # imagem parada

Teclas: 'q' sai | 'w'/'s' sobe/desce o corte do topo | 'a'/'d' ajusta o branco
"""

import sys
import cv2

import config
from vision import LaneVision


vision_ref = [None]


def desenhar_hud(img, res):
    lado = res.lado_continuo or "?"
    txt = (f"erro {res.error:+.2f}  L{int(res.found_left)} R{int(res.found_right)}"
           f"  pts {res.n_pontos}  branco {vision_ref[0].white_frac*100:.0f}%"
           f"  continua={lado}")
    cor = (0,255,255) if res.valid else (0,0,255)
    cv2.putText(img, txt, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cor, 1)
    if not res.valid:
        cv2.putText(img, f"PERDIDO: {res.motivo}", (8, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1)
    return img


def tratar_tecla(k):
    """Ajuste ao vivo. No fim, copie os valores pro config.py."""
    if k == ord('w'):
        config.SCAN_TOP_PCT = min(0.8, config.SCAN_TOP_PCT + 0.02)
    elif k == ord('s'):
        config.SCAN_TOP_PCT = max(0.0, config.SCAN_TOP_PCT - 0.02)
    elif k == ord('d'):
        config.TOPHAT_FATOR = min(1.5, config.TOPHAT_FATOR + 0.05)
    elif k == ord('a'):
        config.TOPHAT_FATOR = max(0.05, config.TOPHAT_FATOR - 0.05)
    elif k == ord('q'):
        return False
    return True


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else 0
    vision = LaneVision()
    vision_ref[0] = vision

    # ---- imagem parada
    if isinstance(src, str) and src.lower().endswith(('.jpg', '.jpeg', '.png')):
        frame = cv2.resize(cv2.imread(src), config.CAMERA_RES)
        while True:
            res = vision.process(frame, debug=True)
            cv2.imshow("Visao", desenhar_hud(res.debug_view, res))
            cv2.imshow("Mascara", res.debug_mask)
            if not tratar_tecla(cv2.waitKey(0) & 0xFF):
                break
        cv2.destroyAllWindows()
        return

    # ---- camera ou video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("Camera/arquivo nao abriu.")
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, config.CAMERA_RES)
        res = vision.process(frame, debug=True)

        cv2.imshow("Original", frame)
        cv2.imshow("Visao", desenhar_hud(res.debug_view, res))
        cv2.imshow("Mascara", res.debug_mask)

        if not tratar_tecla(cv2.waitKey(1) & 0xFF):
            break

    print(f"\nCole no config.py:\n  SCAN_TOP_PCT   = {config.SCAN_TOP_PCT:.2f}"
          f"\n  TOPHAT_FATOR   = {config.TOPHAT_FATOR:.2f}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
