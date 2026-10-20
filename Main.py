"""
Main.py - loop principal do piloto autonomo.

    python3 Main.py            # anda
    python3 Main.py --ver      # anda e mostra as cameras na tela do Pi
                               # (so no terminal da AREA DE TRABALHO do Pi)
    python3 Main.py --sem-log  # nao grava a pasta logs/
Sai: 'q' na janela (--ver) ou Ctrl+C.

Este arquivo SO faz entrada/saida: le as duas cameras, entrega as imagens pro
Piloto (piloto.py, que decide) e manda (speed, turn) pro motor. Nao sabe
processar imagem nem calcular direcao.

  PISTA (config.CAMERA_PISTA_ID): olha pro chao, segue as faixas.
  CONE  (config.CAMERA_CONE_ID):  mais reta, so enxerga cones.
"""

import os
import sys
import time

import cv2
import numpy as np

import config
from MotorModule import Motor
from camera import abrir_camera, CameraThread
from manobra import SEGUIR
from piloto import Piloto
from registro import Registro

# --ver: janelas das cameras na tela do Pi ENQUANTO o carro anda. Custa FPS,
# entao so atualiza a cada config.DEBUG_CADA quadros.
DEBUG = "--ver" in sys.argv
GRAVAR = "--sem-log" not in sys.argv


def main():
    global DEBUG
    print("Piloto autonomo: navegando ENTRE as faixas.")
    if DEBUG and not os.environ.get("DISPLAY"):
        # terminal remoto (so texto): imshow derruba o programa (Qt/xcb)
        print("Sem tela neste terminal (DISPLAY vazio): janelas desligadas. "
              "Rode pelo terminal da area de trabalho do Pi.")
        DEBUG = False

    cap = abrir_camera(config.CAMERA_PISTA_ID, config.CAMERA_RES, "pista")
    if not cap.isOpened():
        print("Erro critico: camera da pista nao abriu.")
        return

    # camera de cone: se nao abrir, o carro segue so a faixa
    cone_cam = None
    if config.CONE_CAMERA_ATIVA:
        cap_cone = abrir_camera(config.CAMERA_CONE_ID, config.CAMERA_CONE_RES, "cone",
                                fps=config.CAMERA_CONE_FPS)
        if cap_cone.isOpened():
            cone_cam = CameraThread(cap_cone)
        else:
            cap_cone.release()
            print("AVISO: camera de cone nao abriu -> SEM desvio de cones.")

    piloto = Piloto()
    motor = Motor(config.ENA_A, config.IN1_A, config.IN2_A,
                  config.ENA_B, config.IN3_B, config.IN4_B,
                  config.SERVO_PIN)
    registro = Registro() if GRAVAR else None

    cone_idx = -1
    n_total = 0
    frames, t_fps = 0, time.time()
    t_cam = t_vis = t_mot = 0.0    # acumuladores pra achar o gargalo real
    fps_medido = 30.0              # otimista ate a 1a medicao real

    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            t1 = time.time()
            if not ok:
                motor.stop()
                print("Frame perdido (checa a alimentacao do Pi).")
                continue
            frame = cv2.resize(frame, config.CAMERA_RES)

            # camera de cone: o frame mais novo SEM esperar (thread). Quadro
            # velho = ignora cone.
            f_cone, cone_novo = None, False
            if cone_cam is not None:
                f, idx, idade = cone_cam.read()
                if f is not None and idade <= config.CONE_FRAME_VELHO_S:
                    f_cone = f
                    if idx != cone_idx:
                        cone_novo, cone_idx = True, idx

            # --ver: desenhar custa FPS. So a cada DEBUG_CADA quadros.
            n_total += 1
            mostrar = DEBUG and (n_total % config.DEBUG_CADA == 0)
            piloto.debug_pista = mostrar

            agora = time.time()
            turn, speed = piloto.step(agora, frame, f_cone, cone_novo,
                                      fps=fps_medido, debug_cone=DEBUG)
            t2 = time.time()
            motor.move(speed, turn)
            t3 = time.time()
            t_cam += (t1 - t0); t_vis += (t2 - t1); t_mot += (t3 - t2)

            for linha in piloto.eventos:
                print("  " + linha)
                if registro:
                    registro.evento(linha)
            if registro:
                registro.telemetria(piloto, fps_medido)
                est = piloto.mudou_estado
                if est is not None:
                    tag = ("IGNORADO" if est == SEGUIR and piloto.manobra.ignorando
                           else est)
                    registro.foto(tag, frame, f_cone, piloto)

            # ---------- telemetria (a cada 2 s) ----------
            frames += 1
            if time.time() - t_fps >= 2.0:
                fps_medido = frames / (time.time() - t_fps)
                _imprime_resumo(piloto, cone_cam, fps_medido, frames, t_cam, t_vis, t_mot)
                frames, t_fps = 0, time.time()
                t_cam = t_vis = t_mot = 0.0

            if mostrar and not _mostra(piloto):
                break

    except KeyboardInterrupt:
        print("\nParada de emergencia.")
    finally:
        motor.cleanup()
        cap.release()
        if cone_cam is not None:
            cone_cam.release()
        if registro:
            registro.fechar()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass                       # OpenCV sem janelas (headless)
        print("Desligado com seguranca.")


def _imprime_resumo(piloto, cone_cam, fps, frames, t_cam, t_vis, t_mot):
    res, cone, desvio, pid = piloto.res, piloto.cone, piloto.desvio, piloto.pid
    # status honesto: "OK" so quando a leitura E fresca de verdade (res.valid
    # fica True no periodo de graca mesmo em frame degradado)
    if res.motivo == "ok":
        status = "OK"
    elif res.valid:
        status = f"DEGRADADO({res.motivo})"
    else:
        status = f"PERDIDO({res.motivo})"
    freio = " [FREANDO por FPS baixo]" if fps < config.FPS_MINIMO_SEGURO else ""
    print(f"FPS {fps:.1f} | erro {res.error:+.2f} | turn {piloto.turn:+.2f} | "
          f"pts {res.n_pontos} | continua={res.lado_continuo or '?'} | {status}{freio}")
    print(f"  termos -> P {pid.last_p:+.2f}  I {pid.last_i:+.2f}  "
          f"D {pid.last_d:+.2f}  FF {pid.last_ff:+.2f}")
    if cone_cam is not None:
        if cone is not None and cone.found:
            cinfo = (f"VISTO x {cone.cx:+.2f} lat {cone.lat:+.1f}w "
                     f"prox {cone.proximidade:.2f} alt {cone.altura:.2f}")
        else:
            cinfo = "nao visto"
        lado_t = {-1: "passa pela DIREITA", 1: "passa pela ESQUERDA", 0: "-"}[desvio.lado]
        print(f"  cone    -> {cinfo} | estado {piloto.manobra.estado} | "
              f"desvio {desvio.offset:+.2f} | {lado_t}")
    print(f"  gargalo -> camera {1000*t_cam/frames:.0f}ms  "
          f"visao+decisao {1000*t_vis/frames:.0f}ms  motor {1000*t_mot/frames:.0f}ms  "
          f"(por frame)")


def _mostra(piloto):
    """Janelas do --ver. Devolve False se apertou 'q'."""
    res, cone, desvio = piloto.res, piloto.cone, piloto.desvio
    if res.debug_view is not None:
        pista_v = np.vstack([res.debug_view, res.debug_mask])
        cv2.putText(pista_v, f"{piloto.manobra.estado}  vel {piloto.speed:.2f}  "
                    f"turn {piloto.turn:+.2f}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 2)
        cv2.imshow("PISTA", pista_v)
    if cone is not None and cone.debug_view is not None:
        cone_v = cone.debug_view.copy()
        lado_t = {-1: "DIREITA", 1: "ESQUERDA", 0: "-"}[desvio.lado]
        pos = (f" cone {desvio.w_cone_cm:+.0f}cm" if desvio.w_cone_cm is not None else "")
        cv2.putText(cone_v, f"{piloto.manobra.estado} lado {lado_t}{pos}",
                    (6, cone_v.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 2)
        cv2.imshow("CONE", cone_v)
    return (cv2.waitKey(1) & 0xFF) != ord('q')


if __name__ == '__main__':
    main()
