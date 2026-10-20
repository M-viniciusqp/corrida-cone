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

import os
import os
import sys
import time
import cv2

import config
from vision import LaneVision
from controller import SteeringPID, speed_for
from MotorModule import Motor
from camera import abrir_camera, CameraThread
from cones import ConeVision, ConeAvoider
import numpy as np
from manobra import Manobra, SEGUIR, FREAR, ANALISAR, DESVIAR

# DEBUG: janelas das cameras na tela do Pi ENQUANTO o carro anda.
#   python3 Main.py --ver     (precisa do terminal da AREA DE TRABALHO do Pi)
# Sem --ver fica desligado = mais FPS = reacao mais rapida.
DEBUG = "--ver" in sys.argv


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

    # camera de cone: se nao abrir, o carro segue so a faixa (como antes)
    cone_cam = None
    if config.CONE_CAMERA_ATIVA:
        cap_cone = abrir_camera(config.CAMERA_CONE_ID, config.CAMERA_CONE_RES, "cone",
                                 fps=config.CAMERA_CONE_FPS)
        if cap_cone.isOpened():
            cone_cam = CameraThread(cap_cone)
        else:
            cap_cone.release()
            print("AVISO: camera de cone nao abriu -> SEM desvio de cones.")

    vision = LaneVision()
    cones = ConeVision()
    desvio = ConeAvoider()
    manobra = Manobra()
    estado_ant = SEGUIR
    peq_ant = None                 # imagem pequena da pista (mede se o carro ainda anda)
    t_man_log = 0.0
    cone = None                    # ultimo ConeResult (so reprocessa frame novo)
    cone_idx = -1
    pid = SteeringPID()
    motor = Motor(config.ENA_A, config.IN1_A, config.IN2_A,
                  config.ENA_B, config.IN3_B, config.IN4_B,
                  config.SERVO_PIN)

    t_cone_log = 0.0               # log rapido do desvio (so com cone ativo)
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
            # durante o desvio o carro cruza a pontilhada: nao deixa a visao
            # "aprender" qual lado e a continua com a imagem desse momento
            vision.congelar_lado = desvio.ativo or desvio.faixa_outra
            res = vision.process(frame, debug=DEBUG)

            # quanto a imagem da pista mudou desde o quadro anterior: com o
            # carro parado fica ~0 (so ruido); andando, o chao passa
            peq = cv2.cvtColor(cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA),
                               cv2.COLOR_BGR2GRAY).astype(np.int16)
            mov_pista = (float(np.abs(peq - peq_ant).mean())
                         if peq_ant is not None else None)
            peq_ant = peq

            # camera de cone: pega o frame mais novo SEM esperar (thread).
            # So reprocessa se chegou frame novo; frame velho = ignora cone.
            cone_novo = False
            if cone_cam is not None:
                f_cone, idx, idade = cone_cam.read()
                if f_cone is None or idade > config.CONE_FRAME_VELHO_S:
                    cone = None
                elif idx != cone_idx:
                    # de que lado esta a faixa CONTINUA (a regra do desvio
                    # mede a distancia do cone ate ela, na imagem do cone)
                    sc = res.sinal_continuo
                    lado_s = 1 if sc > 0.5 else (-1 if sc < -0.5 else 0)
                    cone = cones.process(f_cone, debug=DEBUG, lado_solida=lado_s)
                    cone_idx = idx
                    cone_novo = True
            t2 = time.time()

            # ---------- PARA -> ANALISA -> DESVIA (manobra.py) ----------
            agora = time.time()
            estado = manobra.update(agora, cone, cone_novo, mov_pista, desvio)
            if estado != estado_ant:
                if estado == FREAR:
                    print(f"  [manobra] CONE a vista (alt {cone.altura:.2f}) -> PARANDO...")
                elif estado == ANALISAR:
                    print(f"  [manobra] parou em {agora - manobra.t_ref:.2f}s -> ANALISANDO o cone")
                elif estado == DESVIAR:
                    lado_t = {-1: "pela DIREITA", 1: "pela ESQUERDA", 0: "?"}[desvio.lado]
                    pos = (f"cone {desvio.w_cone_cm:+.0f}cm do centro da faixa"
                           if desvio.w_cone_cm is not None else "posicao nao medida")
                    print(f"  [manobra] DESVIANDO {lado_t} ({pos})")
                elif estado == SEGUIR:
                    print("  [manobra] desvio terminado -> seguindo a faixa")
                if estado_ant in (FREAR, ANALISAR) and estado not in (FREAR, ANALISAR):
                    pid.marcar_retomada()      # saindo da parada: sem chute do D
                    pid.reset()
                estado_ant = estado
            if estado == FREAR and agora - t_man_log >= 0.15:
                t_man_log = agora
                mc = manobra.mov_cone
                mc_t = f"cone d_alt {mc[0]:.3f} d_x {mc[1]:.3f}" if mc else "cone -"
                mp_t = f"{manobra.mov_pista:.1f}" if manobra.mov_pista is not None else "-"
                print(f"  [manobra] freando... {mc_t} | chao {mp_t} "
                      f"(parado = d_alt<{config.CONE_PARADO_D_ALT} d_x<{config.CONE_PARADO_D_X} "
                      f"chao<{config.CONE_PARADO_DIFF}) calmo {manobra.calmo}/{config.CONE_PARADA_CALMO_N}")

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

                # O PID continua vendo SO o erro da pista (D e filtros como
                # antes). O desvio de cone entra DEPOIS como esterco extra
                # proporcional: assim o degrau do cone nao dispara o termo D.
                # O desvio e calculado pra ser um PISO sobre o erro da pista
                # (ver cones.py), entao a pista nao empurra o carro pro cone.
                offset = desvio.update(cone, res)
                if config.CONE_OFFSET_NO_PID or not desvio.modo_pos:
                    turn = pid.update(res.error, res.heading)
                    if offset:
                        adj = desvio.ganho_esterco(config.PID_KP) * offset
                        if config.STEER_INVERT:
                            adj = -adj
                        turn = max(-1.0, min(1.0, turn + adj))
                else:
                    # modo posicao: o deslocamento move o ALVO e passa pelo
                    # PID (o termo D amortece, sem passar do ponto)
                    base = res.error_u if config.CONE_ERRO_SEM_SATURAR else res.error
                    erro = max(-1.0, min(1.0, base +
                                         offset * config.CONE_GANHO_POS))
                    # NUNCA vira pro lado do cone que esta passando: se o carro
                    # passou do alvo (erro da pista saturado), a soma pista +
                    # desvio fica positiva e ele voltava pra cima do cone.
                    if config.CONE_TRAVA_LADO_CONE and desvio.cone_do_lado(cone):
                        if desvio.lado > 0:
                            erro = min(erro, 0.0)
                        else:
                            erro = max(erro, 0.0)
                    turn = pid.update(erro, res.heading)
                    if (manobra.estado == DESVIAR and
                            abs(desvio.offset - desvio.off_base) > 0.1):
                        # desviando: direcao mais forte (o seguidor de faixa sozinho
                        # chega devagar no alvo e nao da tempo antes do cone)
                        turn = max(-1.0, min(1.0, turn * config.CONE_GANHO_TURN_DESVIO))
                speed = speed_for(turn, res.curvature) * desvio.fator_velocidade()
                if desvio.parar:
                    speed = 0.0     # cone colado na frente: nao bate
                if manobra.parado:
                    speed, turn = 0.0, 0.0   # parando/analisando: motores em zero

                # log RAPIDO enquanto desvia (o resumo de 2s e lento demais pra
                # ver o que aconteceu num desvio que dura ~1s)
                if desvio.ativo and time.time() - t_cone_log >= 0.15:
                    t_cone_log = time.time()
                    if cone is not None and cone.found:
                        v = (f"x {cone.cx:+.2f} lat {cone.lat:+.1f}w "
                             f"alt {cone.altura:.2f} prox {cone.proximidade:.2f}")
                    else:
                        v = "fora da visao (segurando)"
                    aviso = "  <<< PARADO: cone na frente" if desvio.parar else ""
                    if desvio.dist_cont_cm is not None:
                        onde = ("na SUA faixa" if desvio.na_faixa
                                else ("na OUTRA faixa" if desvio.na_faixa is False
                                      else "em cima da pontilhada"))
                        aviso = (f" | {desvio.dist_cont_cm:.0f}cm da continua "
                                 f"({onde})") + aviso
                    else:
                        aviso = " | continua nao vista" + aviso
                    if desvio.w_cone_cm is not None:
                        # onde o carro ACHA que o cone esta (cm do centro da
                        # faixa direita; negativo = lado da pontilhada) e pra
                        # onde mira (alvo)
                        aviso += (f" | cone {desvio.w_cone_cm:+.0f}cm "
                                  f"alvo {desvio.w_alvo_cm:+.0f}cm")
                    print(f"  [desvio] {v} | pista {res.error_u:+.2f} "
                          f"desvio {offset:+.2f} -> turn {turn:+.2f} "
                          f"vel {speed:.2f} | lado {desvio.lado:+d}{aviso}")

                # TRAVA DE SEGURANCA: se o FPS medido estiver abaixo do minimo
                # seguro pra velocidade configurada, o codigo reduz sozinho.
                # E o que teria evitado o "saiu da pista" com camera a 6 FPS
                # e VELOCIDADE_BASE=0.80 - 0.80 exige FPS alto, ponto fixo.
                if fps_medido < config.FPS_MINIMO_SEGURO:
                    fator_freio = max(0.35, fps_medido / config.FPS_MINIMO_SEGURO)
                    speed *= fator_freio
            elif vision.lost_frames <= (config.MAX_FRAMES_PERDIDO_DESVIO
                                        if vision.em_desvio
                                        else config.MAX_FRAMES_PERDIDO):
                # perdeu agora (ou leitura degradada): NAO congela o esterco
                # velho (isso trava o carro numa curva errada). Decai em
                # direcao ao reto, bem devagar.
                frame_anterior_confiavel = False
                if manobra.parado:
                    turn, speed = 0.0, 0.0
                elif vision.em_desvio:
                    # no meio de um desvio perder a leitura e esperado: segura
                    # o esterco (decai bem devagar) e nao reduz tanto
                    turn *= config.TURN_DECAI_PERDIDO_DESVIO
                    speed = config.VELOCIDADE_PERDIDO_DESVIO
                else:
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
                        cinfo = (f"VISTO x {cone.cx:+.2f} lat {cone.lat:+.1f}w "
                                 f"prox {cone.proximidade:.2f} alt {cone.altura:.2f}")
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

            if DEBUG:
                if res.debug_view is not None:
                    pista_v = np.vstack([res.debug_view, res.debug_mask])
                    cv2.putText(pista_v, f"{manobra.estado}  vel {speed:.2f}  turn {turn:+.2f}",
                                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0, 255, 255), 2)
                    cv2.imshow("PISTA", pista_v)
                if cone is not None and cone.debug_view is not None:
                    cone_v = cone.debug_view.copy()
                    lado_t = {-1: "DIREITA", 1: "ESQUERDA", 0: "-"}[desvio.lado]
                    pos = (f" cone {desvio.w_cone_cm:+.0f}cm"
                           if desvio.w_cone_cm is not None else "")
                    cv2.putText(cone_v, f"{manobra.estado} lado {lado_t}{pos}",
                                (6, cone_v.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 255), 2)
                    cv2.imshow("CONE", cone_v)
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

