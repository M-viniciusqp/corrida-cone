"""
piloto.py - o CEREBRO do carro: recebe as imagens, devolve (turn, speed).

Nao conhece camera, motor nem GPIO. Por isso o MESMO codigo roda no carro
(Main.py) e no simulador (sim/closedloop.py): o que o simulador testa e o que
anda na pista.

    Main.py / simulador
          |  imagens + hora
          v
      Piloto.step()
          |-- vision.py   (faixa: erro, curvatura, linhas)
          |-- cones.py    (ConeVision: acha o cone | ConeAvoider: posicao e lado)
          |-- manobra.py  (para -> analisa -> desvia; ignora cone de outra faixa)
          |-- controller.py (PID + velocidade)
          v
      (turn, speed)  +  self.eventos (linhas de log)  +  self.res / self.cone (HUD)
"""

import cv2
import numpy as np

import config
from vision import LaneVision
from controller import SteeringPID, speed_for
from cones import ConeVision, ConeAvoider
from manobra import Manobra, SEGUIR, FREAR, ANALISAR, DESVIAR


class Piloto:
    def __init__(self):
        self.vision = LaneVision()
        self.cones = ConeVision()
        self.desvio = ConeAvoider()
        self.manobra = Manobra()
        self.pid = SteeringPID()

        self.res = None             # ultimo LaneResult
        self.cone = None            # ultimo ConeResult (so reprocessa quadro novo)
        self.turn = 0.0
        self.speed = 0.0
        self.eventos = []           # linhas de log geradas no ultimo step()
        self.mudou_estado = None    # estado novo da manobra neste step (pra foto do evento)
        self.debug_pista = False    # pede as imagens de debug da visao (--ver)

        self._confiavel_ant = True
        self._peq_ant = None        # imagem pequena da pista (mede se o carro anda)
        self._estado_ant = SEGUIR
        self._t_log_desvio = 0.0
        self._t_log_freio = 0.0

    # ------------------------------------------------------------------
    def step(self, agora, frame_pista, frame_cone=None, cone_novo=False,
             fps=30.0, debug_cone=False):
        """frame_pista: BGR no tamanho config.CAMERA_RES.
        frame_cone: ultimo quadro da camera de cone (None = sem camera ou quadro
        velho -> sem cone). cone_novo: o quadro e novo desde o ultimo step.
        fps: FPS medido do laco (freio de seguranca)."""
        self.eventos = []
        self.mudou_estado = None
        desvio, manobra, pid = self.desvio, self.manobra, self.pid

        # durante o desvio o carro cruza a pontilhada: nao deixa a visao
        # "aprender" qual lado e a continua com a imagem desse momento
        self.vision.congelar_lado = desvio.ativo or desvio.faixa_outra
        self.vision.medir_cone = manobra.estado in (FREAR, ANALISAR)
        res = self.vision.process(frame_pista, debug=self.debug_pista)
        self.res = res

        # quanto a imagem da pista mudou desde o quadro anterior: com o carro
        # parado fica ~0 (so ruido); andando, o chao passa
        peq = cv2.cvtColor(cv2.resize(frame_pista, (64, 36), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2GRAY).astype(np.int16)
        mov_pista = (float(np.abs(peq - self._peq_ant).mean())
                     if self._peq_ant is not None else None)
        self._peq_ant = peq

        # cone: so reprocessa quadro novo
        if frame_cone is None:
            self.cone = None
        elif cone_novo:
            sc = res.sinal_continuo
            lado_s = 1 if sc > 0.5 else (-1 if sc < -0.5 else 0)
            self.cone = self.cones.process(frame_cone, debug=debug_cone, lado_solida=lado_s)
        cone = self.cone

        self._manobra(agora, cone, cone_novo, mov_pista, res)
        turn, speed = self._controle(agora, res, cone, fps)
        self.turn, self.speed = turn, speed
        return turn, speed

    # ------------------------------------------------------------------
    def _manobra(self, agora, cone, cone_novo, mov_pista, res):
        desvio, manobra = self.desvio, self.manobra
        estado = manobra.update(agora, cone, cone_novo, mov_pista, desvio, res)
        ev = self.eventos
        if estado != self._estado_ant:
            self.mudou_estado = estado
            if estado == FREAR:
                ev.append(f"[manobra] CONE a vista (alt {cone.altura:.2f}) -> PARANDO...")
            elif estado == ANALISAR:
                ev.append(f"[manobra] parou em {agora - manobra.t_ref:.2f}s -> ANALISANDO o cone")
            elif estado == SEGUIR and manobra.ignorando:
                ev.append(f"[manobra] cone NAO esta na minha faixa -> segue em frente "
                          f"({manobra.ultimo})")
            elif estado == DESVIAR:
                lado_t = {-1: "pela DIREITA", 1: "pela ESQUERDA", 0: "?"}[desvio.lado]
                pos = (f"cone {desvio.w_cone_cm:+.0f}cm do centro da faixa"
                       if desvio.w_cone_cm is not None else "posicao nao medida")
                ev.append(f"[manobra] DESVIANDO {lado_t} ({pos}) | cone na MINHA faixa: "
                          f"{manobra.ultimo}")
            elif estado == SEGUIR:
                ev.append("[manobra] desvio terminado -> seguindo a faixa")
            if self._estado_ant in (FREAR, ANALISAR) and estado not in (FREAR, ANALISAR):
                self.pid.marcar_retomada()      # saindo da parada: sem chute do D
                self.pid.reset()
            self._estado_ant = estado
        if estado == FREAR and agora - self._t_log_freio >= 0.15:
            self._t_log_freio = agora
            mc = manobra.mov_cone
            mc_t = f"cone d_alt {mc[0]:.3f} d_x {mc[1]:.3f}" if mc else "cone -"
            mp_t = f"{manobra.mov_pista:.1f}" if manobra.mov_pista is not None else "-"
            ev.append(f"[manobra] freando... {mc_t} | chao {mp_t} "
                      f"(parado = d_alt<{config.CONE_PARADO_D_ALT} d_x<{config.CONE_PARADO_D_X} "
                      f"chao<{config.CONE_PARADO_DIFF}) calmo {manobra.calmo}/{config.CONE_PARADA_CALMO_N}")

    # ------------------------------------------------------------------
    def _controle(self, agora, res, cone, fps):
        desvio, manobra, pid, vision = self.desvio, self.manobra, self.pid, self.vision
        turn, speed = self.turn, self.speed

        # IMPORTANTE: checa res.motivo=="ok", nao so res.valid. res.valid fica
        # True durante o "periodo de graca" mesmo em frames de baixa confianca,
        # onde o erro fica CONGELADO no ultimo valor bom: o PID veria um salto
        # violento quando chegasse um quadro fresco. So roda com leitura fresca.
        if res.valid and res.motivo == "ok" and (res.found_left or res.found_right):
            if not self._confiavel_ant:
                pid.marcar_retomada()
            self._confiavel_ant = True

            offset = desvio.update(cone, res, agora)
            # o desvio move o ALVO e passa pelo PID (o termo D amortece, sem
            # passar do ponto). Usa o erro SEM o corte em +-1: o corte escondia o
            # quanto o carro passou do alvo.
            base = res.error_u if config.CONE_ERRO_SEM_SATURAR else res.error
            erro = max(-1.0, min(1.0, base + offset * config.CONE_GANHO_POS))
            # NUNCA vira pro lado do cone que esta passando
            if config.CONE_TRAVA_LADO_CONE and desvio.cone_do_lado(cone):
                erro = min(erro, 0.0) if desvio.lado > 0 else max(erro, 0.0)
            turn = pid.update(erro, res.heading)
            if (manobra.estado == DESVIAR and
                    abs(desvio.offset - desvio.off_base) > 0.1):
                turn = max(-1.0, min(1.0, turn * config.CONE_GANHO_TURN_DESVIO))

            speed = speed_for(turn, res.curvature) * desvio.fator_velocidade()
            if manobra.parado:
                speed, turn = 0.0, 0.0           # parando/analisando: motores em zero
            elif (manobra.estado == DESVIAR and
                    agora - manobra.t_estado < config.CONE_ARRANQUE_S):
                # arrancando do repouso: o motor nao vence o atrito na velocidade
                # baixa e o desvio "demora" pra comecar. Da um empurrao curto.
                speed = max(speed, config.CONE_ARRANQUE_VEL)

            self._log_desvio(agora, res, cone, offset, turn, speed)

            # TRAVA DE SEGURANCA: FPS abaixo do minimo seguro -> reduz sozinho
            if fps < config.FPS_MINIMO_SEGURO:
                speed *= max(0.35, fps / config.FPS_MINIMO_SEGURO)

        elif vision.lost_frames <= (config.MAX_FRAMES_PERDIDO_DESVIO
                                    if vision.em_desvio else config.MAX_FRAMES_PERDIDO):
            # perdeu agora (ou leitura degradada): NAO congela o esterco velho,
            # decai em direcao ao reto, devagar.
            self._confiavel_ant = False
            if manobra.parado:
                turn, speed = 0.0, 0.0
            elif vision.em_desvio:
                turn *= config.TURN_DECAI_PERDIDO_DESVIO
                speed = config.VELOCIDADE_PERDIDO_DESVIO
            else:
                turn *= 0.85
                speed = config.VELOCIDADE_PERDIDO
        else:
            turn, speed = 0.0, 0.0
            pid.reset()
            desvio.reset()
        return turn, speed

    def _log_desvio(self, agora, res, cone, offset, turn, speed):
        """Log RAPIDO enquanto desvia (o resumo de 2 s e lento demais)."""
        desvio = self.desvio
        if not desvio.ativo or agora - self._t_log_desvio < 0.15:
            return
        self._t_log_desvio = agora
        if cone is not None and cone.found:
            v = (f"x {cone.cx:+.2f} lat {cone.lat:+.1f}w "
                 f"alt {cone.altura:.2f} prox {cone.proximidade:.2f}")
        else:
            v = "fora da visao (segurando)"
        aviso = ""
        if desvio.w_cone_cm is not None:
            aviso = (f" | cone {desvio.w_cone_cm:+.0f}cm alvo {desvio.w_alvo_cm:+.0f}cm "
                     f"[{desvio.fonte_w}]")
        self.eventos.append(
            f"[desvio] {v} | pista {res.error_u:+.2f} desvio {offset:+.2f} -> "
            f"turn {turn:+.2f} vel {speed:.2f} | lado {desvio.lado:+d}{aviso}")
