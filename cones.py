"""
cones.py - deteccao de cones (camera frontal) e desvio.

Duas pecas, igual o resto do projeto (cada uma faz uma coisa):

  ConeVision   : frame da camera de cone -> ConeResult (onde esta o cone mais
                 perigoso e o quao perto ele esta). Nao sabe nada de motor.

  ConeAvoider  : ConeResult + leitura da pista -> um DESLOCAMENTO somado ao
                 erro lateral. Nao troca o controle: o carro continua seguindo
                 a faixa pelo mesmo PID, so que com o alvo empurrado pro lado
                 contrario ao cone.

Por que somar no erro em vez de um modo "desvio" separado:
  - o PID, o filtro de derivada e o freio de curva continuam valendo;
  - o desvio e sempre LIMITADO pelas bordas da pista (vision.lane_center_x e
    vision.hw_px), entao o carro nunca e empurrado pra fora da faixa pra fugir
    de um cone;
  - se a camera de cone falhar, o carro simplesmente volta a ser o piloto de
    faixa de antes.

Convencao de sinais (a mesma do erro da pista):
  x do cone: -1 = borda esquerda da imagem, +1 = borda direita.
  offset > 0 = empurra o alvo pra DIREITA (carro vira a direita).
"""

import time
import cv2
import numpy as np
import config


# ======================================================================
# DETECCAO
# ======================================================================
class ConeResult:
    __slots__ = ("found", "cx", "altura", "proximidade", "bbox",
                 "n_candidatos", "debug_view", "debug_mask")

    def __init__(self):
        self.found = False
        self.cx = 0.0             # -1..+1, centro do cone na imagem
        self.altura = 0.0         # altura do cone / altura da imagem
        self.proximidade = 0.0    # 0 (longe) .. 1 (colado)
        self.bbox = None          # (x, y, w, h) em pixels do frame
        self.n_candidatos = 0
        self.debug_view = None
        self.debug_mask = None


class ConeVision:
    def __init__(self):
        self._k_open = np.ones((3, 3), np.uint8)
        # fecha os buracos que o brilho (reflexo) do plastico abre no cone
        self._k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

    def mask(self, roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        s, v = config.CONE_S_MIN, config.CONE_V_MIN
        m1 = cv2.inRange(hsv, (0, s, v), (config.CONE_H_BAIXO, 255, 255))
        m2 = cv2.inRange(hsv, (config.CONE_H_ALTO, s, v), (179, 255, 255))
        m = cv2.bitwise_or(m1, m2)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self._k_open)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self._k_close)
        return m

    def process(self, frame, debug=False):
        res = ConeResult()
        H, W = frame.shape[:2]
        y0 = int(H * config.CONE_ROI_TOP_PCT)
        roi = frame[y0:, :]
        m = self.mask(roi)

        # [-2] funciona no OpenCV 3 (3 retornos) e no 4 (2 retornos)
        contornos = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)[-2]
        area_min = config.CONE_AREA_MIN_PCT * W * H

        melhor = None
        for c in contornos:
            area = cv2.contourArea(c)
            if area < area_min:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w == 0 or h == 0:
                continue
            aspecto = h / w
            if not (config.CONE_ASPECTO_MIN <= aspecto <= config.CONE_ASPECTO_MAX):
                continue
            if area / (w * h) < config.CONE_PREENCHIMENTO_MIN:
                continue
            res.n_candidatos += 1
            # o mais ALTO na imagem = o mais perto = o que importa
            if melhor is None or h > melhor[3]:
                melhor = (x, y + y0, w, h)

        if melhor is not None:
            x, y, w, h = melhor
            res.found = True
            res.bbox = melhor
            res.cx = ((x + w / 2.0) - W / 2.0) / (W / 2.0)
            res.altura = h / float(H)

            toca_base = (y + h) >= H - 2      # cortado pela borda de baixo
            if toca_base:
                res.proximidade = 1.0
            else:
                a, b = config.CONE_ALTURA_LONGE, config.CONE_ALTURA_PERTO
                res.proximidade = float(np.clip((res.altura - a) / (b - a), 0.0, 1.0))

        if debug:
            res.debug_view = self._draw(frame, res)
            res.debug_mask = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        return res

    @staticmethod
    def _draw(frame, res):
        vis = frame.copy()
        H, W = vis.shape[:2]
        cv2.line(vis, (W // 2, 0), (W // 2, H), (255, 255, 255), 1)
        if res.found:
            x, y, w, h = res.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis, f"x {res.cx:+.2f} prox {res.proximidade:.2f}",
                        (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 0), 1)
        return vis


# ======================================================================
# DESVIO
# ======================================================================
class ConeAvoider:
    def __init__(self):
        self.offset = 0.0          # saida suavizada (soma no erro da pista)
        self.lado = 0              # onde o cone fica na imagem: -1 esq, +1 dir
        self.prox = 0.0            # proximidade do cone em uso (p/ freio e HUD)
        self.ativo = False

        self._alvo = 0.0           # ultimo deslocamento "bruto" com cone visivel
        self._t_visto = None       # quando viu cone pela ultima vez
        self._prox_visto = 0.0
        self._cx_ant = None

    def reset(self):
        self.offset = 0.0
        self.lado = 0
        self.prox = 0.0
        self.ativo = False
        self._alvo = 0.0
        self._t_visto = None
        self._prox_visto = 0.0
        self._cx_ant = None

    # ------------------------------------------------------------------
    def update(self, cone, lane, agora=None):
        """cone: ConeResult | None.  lane: LaneResult (ja validado).
        Retorna o deslocamento a somar no erro da pista."""
        agora = time.time() if agora is None else agora

        visivel = (cone is not None and cone.found and
                   abs(cone.cx) < config.CONE_CORREDOR)

        if visivel:
            self._decidir_lado(cone, agora)
            alvo = self._deslocamento(cone) * cone.proximidade * config.CONE_GANHO
            alvo = self._limitar_pela_pista(alvo, lane)
            self._alvo = alvo
            self._t_visto = agora
            self._prox_visto = cone.proximidade
            self._cx_ant = cone.cx
            self.prox = cone.proximidade
            self.ativo = True
        else:
            alvo = self._sem_cone(agora)

        b = config.CONE_OFFSET_SUAVIZACAO
        self.offset = b * self.offset + (1.0 - b) * alvo
        if abs(self.offset) < 0.005 and not self.ativo:
            self.offset = 0.0
        return self.offset

    def fator_velocidade(self):
        """Multiplicador (<=1) da velocidade: desvia mais devagar."""
        if not self.ativo:
            return 1.0
        # so freia se de fato esta desviando (cone fora do caminho nao freia)
        relevancia = min(1.0, abs(self.offset) / 0.15)
        return 1.0 - config.CONE_FREIO_MAX * self.prox * relevancia

    # ------------------------------------------------------------------
    def _decidir_lado(self, cone, agora):
        """Decide por qual lado passa. So re-decide quando e OUTRO cone
        (sumiu por um tempo, ou o x saltou) -- senao fica travado, pra nao
        ficar trocando de lado enquanto o cone esta quase no centro."""
        novo = (self.lado == 0 or self._t_visto is None or
                agora - self._t_visto > config.CONE_RECONHECE_S or
                (self._cx_ant is not None and
                 abs(cone.cx - self._cx_ant) > config.CONE_TROCA_CX))
        if not novo:
            return

        if abs(cone.cx) >= config.CONE_ZONA_CENTRO:
            self.lado = -1 if cone.cx < 0 else 1
        else:
            # cone bem no meio: usa o lado padrao.
            # passa pela direita -> cone fica na ESQUERDA da imagem (-1).
            self.lado = -1 if config.CONE_LADO_PADRAO == "direita" else 1

    @staticmethod
    def _folga_deslocamento(lado, cx):
        # lado -1: cone tem que ficar em x <= -FOLGA  -> empurra pra direita
        # lado +1: cone tem que ficar em x >= +FOLGA  -> empurra pra esquerda
        if lado < 0:
            return max(0.0, cx + config.CONE_FOLGA)
        return min(0.0, cx - config.CONE_FOLGA)

    def _deslocamento(self, cone):
        return self._folga_deslocamento(self.lado, cone.cx)

    def _limitar_pela_pista(self, off, lane):
        """Nunca deixa o alvo final passar da borda da pista (menos margem)."""
        off = max(-config.CONE_OFFSET_MAX, min(config.CONE_OFFSET_MAX, off))
        if lane is None or lane.hw_px is None:
            return off

        metade = config.CAMERA_RES[0] / 2.0
        margem = lane.hw_px * config.CONE_MARGEM_PISTA
        esq = lane.lane_center_x - lane.hw_px + margem
        dire = lane.lane_center_x + lane.hw_px - margem

        # quanto ainda cabe pra cada lado a partir do alvo atual. Se o alvo
        # ja esta fora (vies de faixa), nao empurra MAIS pra fora.
        lo = min(0.0, esq - lane.center_x)
        hi = max(0.0, dire - lane.center_x)
        off_px = max(lo, min(hi, off * metade))
        return off_px / metade

    def _sem_cone(self, agora):
        """Cone sumiu (saiu do campo de visao ou falhou a deteccao)."""
        if self._t_visto is None:
            self.ativo = False
            return 0.0

        dt = agora - self._t_visto
        segurar = (config.CONE_SEGURAR_S if self._prox_visto >= 0.5
                   else config.CONE_SEGURAR_LONGE_S)

        if dt <= segurar:
            return self._alvo               # segura: o carro ainda esta passando

        f = 1.0 - (dt - segurar) / config.CONE_DECAI_S
        if f <= 0.0:
            self._alvo = 0.0
            self._t_visto = None
            self.lado = 0
            self.prox = 0.0
            self.ativo = False
            return 0.0
        return self._alvo * f
