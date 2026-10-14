"""
main.py - loop principal do piloto autonomo.

Roda: python3 main.py
Sai:  tecla 'q' na janela, ou Ctrl+C

Responsabilidade deste arquivo: SO orquestrar.
Ele nao sabe processar imagem (isso e do vision.py), nao sabe calcular
correcao (controller.py) e nao sabe falar com GPIO (MotorModule.py).

Duas cameras:
  - PISTA (config.CAMERA_PISTA_ID): olha pro chao, segue as faixas. Igual a antes.
  - CONE  (config.CAMERA_CONE_ID):  mais reta, so enxerga cones (cones.py).
O cone nao assume o controle: ele soma um deslocamento ao erro da pista.
"""

import time
import cv2

import config
from vision import LaneVision
from controller import SteeringPID, speed_for
from MotorModule import Motor
from camera import abrir_camera, CameraThread
from cones import ConeVision, ConeAvoider

DEBUG = False          # desligado = mais FPS = reacao mais rapida


def main():
    print("Piloto autonomo: navegando ENTRE as faixas.")

    cap = abrir_camera(config.CAMERA_PISTA_ID, config.CAMERA_RES, "pista")
    if not cap.isOpened():
        print("Erro critico: camera da pista nao abriu.")
        return

    # camera de cone: se nao abrir, o carro segue so a faixa (como antes)
    cone_cam = None
    if config.CONE_CAMERA_ATIVA:
        cap_cone = abrir_camera(config.CAMERA_CONE_ID, config.CAMERA_CONE_RES, "cone")
        if cap_cone.isOpened():
            cone_cam = CameraThread(cap_cone)
        else:
            cap_cone.release()
            print("AVISO: camera de cone nao abriu -> SEM desvio de cones.")

    vision = LaneVision()
    cones = ConeVision()
    desvio = ConeAvoider()
    cone = None                    # ultimo ConeResult (so reprocessa frame novo)
    cone_idx = -1
    pid = SteeringPID()
    motor = Motor(config.ENA_A, config.IN1_A, config.IN2_A,
                  config.ENA_B, config.IN3_B, config.IN4_B,
                  config.SERVO_PIN)

    frames, t_fps = 0, time.time()
    turn = 0.0
    t_cam = t_vis = t_mot = 0.0    # acumuladores pra achar o gargalo real
    fps_medido = 30.0              # otimista ate a 1a medicao real
    frame_anterior_confiavel = True

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
            res = vision.process(frame, debug=DEBUG)

            # camera de cone: pega o frame mais novo SEM esperar (thread).
            # So reprocessa se chegou frame novo; frame velho = ignora cone.
            if cone_cam is not None:
                f_cone, idx, idade = cone_cam.read()
                if f_cone is None or idade > config.CONE_FRAME_VELHO_S:
                    cone = None
                elif idx != cone_idx:
                    cone = cones.process(f_cone, debug=DEBUG)
                    cone_idx = idx
            t2 = time.time()

            # ---------- maquina de estados ----------
            # IMPORTANTE: checa res.motivo=="ok", nao so res.valid.
            # res.valid fica True durante o "periodo de graca" mesmo em
            # frames de baixa confianca (ex: poucos_pontos), onde o erro
            # fica CONGELADO no ultimo valor bom. Se o PID rodar em cima
            # desse erro congelado e depois um frame fresco chegar com erro
            # bem diferente, o termo D ve isso como um salto violento e
            # dispara um esterco que nao tem nada a ver com a pista real.
            # Por isso, so roda o PID com leitura fresca de verdade.
            if res.valid and res.motivo == "ok" and (res.found_left or res.found_right):
                if not frame_anterior_confiavel:
                    pid.marcar_retomada()   # evita o chute do D nesta transicao
                frame_anterior_confiavel = True

                # erro da pista + deslocamento do desvio de cone (0 sem cone).
                # O PID, o filtro de D e o freio de curva ficam exatamente
                # como antes; o cone so empurra o alvo pro lado contrario.
                offset = desvio.update(cone, res)
                erro = max(-1.0, min(1.0, res.error + offset))

                turn = pid.update(erro, res.heading)
                speed = speed_for(turn, res.curvature) * desvio.fator_velocidade()

                # TRAVA DE SEGURANCA: se o FPS medido estiver abaixo do minimo
                # seguro pra velocidade configurada, o codigo reduz sozinho.
                # E o que teria evitado o "saiu da pista" com camera a 6 FPS
                # e VELOCIDADE_BASE=0.80 - 0.80 exige FPS alto, ponto fixo.
                if fps_medido < config.FPS_MINIMO_SEGURO:
                    fator_freio = max(0.35, fps_medido / config.FPS_MINIMO_SEGURO)
                    speed *= fator_freio
            elif vision.lost_frames <= config.MAX_FRAMES_PERDIDO:
                # perdeu agora (ou leitura degradada): NAO congela o esterco
                # velho (isso trava o carro numa curva errada). Decai em
                # direcao ao reto, bem devagar.
                frame_anterior_confiavel = False
                turn *= 0.85
                speed = config.VELOCIDADE_PERDIDO
            else:
                turn, speed = 0.0, 0.0
                pid.reset()
                desvio.reset()

            motor.move(speed, turn)
            t3 = time.time()

            t_cam += (t1 - t0); t_vis += (t2 - t1); t_mot += (t3 - t2)

            # ---------- telemetria ----------
            frames += 1
            if time.time() - t_fps >= 2.0:
                dt = time.time() - t_fps
                fps_medido = frames / dt
                # status honesto: "OK" so quando a leitura E fresca de verdade.
                # Antes isso usava so res.valid, que fica True durante o
                # periodo de graca mesmo em frame degradado -- escondia que
                # o carro ja estava em modo de seguranca (decaindo esterco)
                # atras de um rotulo "OK" enganoso.
                if res.motivo == "ok":
                    status = "OK"
                elif res.valid:
                    status = f"DEGRADADO({res.motivo})"
                else:
                    status = f"PERDIDO({res.motivo})"
                freio = " [FREANDO por FPS baixo]" if fps_medido < config.FPS_MINIMO_SEGURO else ""
                lado = res.lado_continuo or "?"
                print(f"FPS {fps_medido:.1f} | erro {res.error:+.2f} | turn {turn:+.2f} | "
                      f"pts {res.n_pontos} | continua={lado} | {status}{freio}")
                print(f"  termos -> P {pid.last_p:+.2f}  I {pid.last_i:+.2f}  "
                      f"D {pid.last_d:+.2f}  FF {pid.last_ff:+.2f}")
                if cone_cam is not None:
                    if cone is not None and cone.found:
                        cinfo = (f"VISTO x {cone.cx:+.2f} prox {cone.proximidade:.2f} "
                                 f"alt {cone.altura:.2f}")
                    else:
                        cinfo = "nao visto"
                    lado_t = {-1: "passa pela DIREITA", 1: "passa pela ESQUERDA",
                              0: "-"}[desvio.lado]
                    print(f"  cone    -> {cinfo} | desvio {desvio.offset:+.2f} "
                          f"| {lado_t}")
                print(f"  gargalo -> camera {1000*t_cam/frames:.0f}ms  "
                      f"visao {1000*t_vis/frames:.0f}ms  motor {1000*t_mot/frames:.0f}ms  "
                      f"(por frame)")
                frames, t_fps = 0, time.time()
                t_cam = t_vis = t_mot = 0.0

            if DEBUG and res.debug_view is not None:
                cv2.imshow("Visao", res.debug_view)
                cv2.imshow("Mascara", res.debug_mask)
                if cone is not None and cone.debug_view is not None:
                    cv2.imshow("Cone", cone.debug_view)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nParada de emergencia.")
    finally:
        motor.cleanup()
        cap.release()
        if cone_cam is not None:
            cone_cam.release()
        cv2.destroyAllWindows()
        print("Desligado com seguranca.")


if __name__ == '__main__':
    main()

