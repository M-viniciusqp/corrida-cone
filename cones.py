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

Como o desvio COMPOE com a pista (importante):
  NAO e erro_pista + desvio. Assim a pista, querendo ir pra esquerda, ainda
  puxava o carro pra cima de um cone que estava a esquerda. O desvio e um
  PISO: o alvo final nunca pode ficar do lado errado do cone.
      cone a esquerda: alvo_final = max(erro_pista, x_cone + folga)
      cone a direita : alvo_final = min(erro_pista, x_cone - folga)
  Se a pista ja quer ir pro lado seguro, o desvio e zero e nao atrapalha.

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
                 "n_candidatos", "debug_view", "debug_mask", "lat", "larg",
                 "toca_base")

    def __init__(self):
        self.found = False
        self.cx = 0.0             # -1..+1, centro do cone na imagem
        self.lat = 0.0            # distancia lateral em LARGURAS DE CONE
        self.larg = 0.0           # largura do cone em unidades de x (-1..+1)
        self.toca_base = False    # base do cone cortada pela borda de baixo
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

            # largura de referencia: se a base do cone saiu do quadro, a
            # largura medida encolhe e a distancia lateral parece MAIOR do
            # que e. Pela altura, estima a largura do cone inteiro.
            w_ref = max(float(w), h / config.CONE_ASPECTO_REAL)
            res.larg = w_ref / (W / 2.0)
            res.lat = ((x + w / 2.0) - W / 2.0) / w_ref

            toca_base = (y + h) >= H - 2      # cortado pela borda de baixo
            res.toca_base = toca_base
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
            cv2.putText(vis, f"x {res.cx:+.2f} lat {res.lat:+.1f}w prox {res.proximidade:.2f}",
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
        self.parar = False         # cone colado na frente: Main zera a velocidade
        self.dist_pont_cm = None   # + = cone a DIREITA da pontilhada (None = n/d)

        self._alvo = 0.0           # ultimo deslocamento "bruto" com cone visivel
        self._t_visto = None       # quando viu cone pela ultima vez
        self._prox_visto = 0.0
        self._cx_ant = None
        self._t_ant = None         # p/ limitar a velocidade de variacao

    def reset(self):
        self.offset = 0.0
        self.lado = 0
        self.prox = 0.0
        self.ativo = False
        self.parar = False
        self.dist_pont_cm = None
        self._alvo = 0.0
        self._t_visto = None
        self._prox_visto = 0.0
        self._cx_ant = None
        self._t_ant = None

    # ------------------------------------------------------------------
    def update(self, cone, lane, agora=None):
        """cone: ConeResult | None.  lane: LaneResult (ja validado).
        Retorna o deslocamento a somar no erro da pista."""
        agora = time.time() if agora is None else agora

        visivel = (cone is not None and cone.found and
                   abs(cone.cx) < config.CONE_CORREDOR)

        self.parar = False
        if visivel:
            e_pista = lane.error if lane is not None else 0.0
            self._decidir_lado(cone, agora, e_pista, lane)
            # a folga ja e geometrica (larguras de cone), entao o pedido
            # cresce sozinho com a aproximacao; a rampa so evita reagir a
            # ruido muito longe.
            rampa = 1.0 if cone.proximidade > 0.0 else 0.0
            alvo = self._deslocamento(cone, e_pista) * rampa * config.CONE_GANHO
            alvo = self._limitar_pela_pista(alvo, lane)

            # rede de seguranca: cone colado BEM na frente -> para
            if (config.CONE_PARAR_ALTURA > 0.0 and
                    abs(cone.cx) < config.CONE_PARAR_X and
                    (cone.altura >= config.CONE_PARAR_ALTURA or cone.toca_base)):
                self.parar = True
            self._alvo = alvo
            self._t_visto = agora
            self._prox_visto = cone.proximidade
            self._cx_ant = cone.cx
            self.prox = cone.proximidade
            self.ativo = True
        else:
            alvo = self._sem_cone(agora)

        # suaviza (EMA) e ainda limita a velocidade de variacao: o cone
        # aparece/some de repente e isso vira degrau no volante.
        dt = 1 / 30.0 if self._t_ant is None else max(0.0, min(0.2, agora - self._t_ant))
        self._t_ant = agora
        b = config.CONE_OFFSET_SUAVIZACAO
        novo = b * self.offset + (1.0 - b) * alvo
        passo = config.CONE_OFFSET_SLEW * dt
        self.offset += max(-passo, min(passo, novo - self.offset))
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
    def _sobras(self, cone, e_pista, lane):
        """Pra cada lado, quanto SOBRA de espaco depois de cumprir a folga
        (negativo = nao cabe). Lado -1 = passa pela DIREITA (offset +),
        lado +1 = passa pela ESQUERDA (offset -)."""
        lo, hi = self._limites(lane)
        folga = config.CONE_FOLGA_LARGURAS * cone.larg
        precisa_dir = max(0.0, cone.cx + folga - e_pista)
        precisa_esq = max(0.0, -(cone.cx - folga - e_pista))
        return {-1: hi - precisa_dir, +1: (-lo) - precisa_esq}

    def _decidir_lado(self, cone, agora, e_pista, lane):
        """Decide por qual lado passa, olhando onde TEM ESPACO.

        Lado preferido: o do cone (cone a esquerda -> passa pela direita).
        Cone no meio -> passa pelo lado da faixa PONTILHADA (e onde pode
        cruzar). Se o preferido nao cabe e o outro cabe, usa o outro.
        Fica travado no mesmo cone; so troca se virar inviavel."""
        sobra = self._sobras(cone, e_pista, lane)
        # a cada frame (alimenta o log mesmo com o lado ja travado)
        pref_pont = self._lado_pela_pontilhada(cone, lane, sobra)

        novo = (self.lado == 0 or self._t_visto is None or
                agora - self._t_visto > config.CONE_RECONHECE_S or
                (self._cx_ant is not None and
                 abs(cone.cx - self._cx_ant) > config.CONE_TROCA_CX))

        if not novo:
            # travado: so troca se o lado atual ficou inviavel e o outro cabe
            # (e so com o cone ainda longe: trocar colado e pior que seguir)
            if (cone.altura < config.CONE_TROCA_LADO_ALTURA_MAX and
                    sobra[self.lado] < -0.05 and sobra[-self.lado] >= 0.0):
                self.lado = -self.lado
            return

        pref = pref_pont
        if pref is None:
            # sem referencia da pontilhada: cai na posicao do cone na imagem
            if abs(cone.cx) >= config.CONE_ZONA_CENTRO:
                pref = -1 if cone.cx < 0 else 1
            else:
                pref = -1 if config.CONE_LADO_PADRAO == "direita" else 1

        if sobra[pref] >= 0.0 or sobra[pref] >= sobra[-pref]:
            self.lado = pref
        else:
            self.lado = -pref

    def _lado_pela_pontilhada(self, cone, lane, sobra):
        """Regra do usuario: de que lado DA PONTILHADA o cone esta?

        Cone dentro da faixa do carro (entre a pontilhada e a continua)
        -> carro cruza a pontilhada.
        Cone na outra faixa -> carro se afasta, indo pro lado da continua.

        Retorna lado (-1 = passa pela direita, +1 = pela esquerda) ou None
        se nao ha referencia confiavel (sem pista/continua indefinida)."""
        self.dist_pont_cm = None
        if lane is None or lane.hw_px is None:
            return None
        s = getattr(lane, "sinal_continuo", 0.0)
        if abs(s) < 0.5:
            return None

        dashed_esq = s > 0                  # continua a direita -> pontilhada a esquerda
        car_x = config.CAMERA_RES[0] * config.CAR_CENTER_PCT
        dashed_x = lane.lane_center_x + (-lane.hw_px if dashed_esq else lane.hw_px)

        # posicao lateral (cm, + = direita do carro) das duas coisas
        pont_cm = (dashed_x - car_x) / lane.hw_px * (config.PISTA_LARGURA_CM / 2.0)
        cone_cm = cone.lat * config.CONE_LARGURA_CM + config.CONE_CAM_OFFSET_CM
        delta = cone_cm - pont_cm           # + = cone a direita da pontilhada
        self.dist_pont_cm = delta

        if abs(delta) < config.CONE_NA_LINHA_CM:
            # em cima da linha: vai pro lado com mais espaco
            return -1 if sobra[-1] >= sobra[+1] else 1

        na_minha_faixa = (delta > 0) if dashed_esq else (delta < 0)
        if na_minha_faixa:
            # cruza a pontilhada: esquerda se ela e a esquerda, senao direita
            return 1 if dashed_esq else -1
        # cone na outra faixa: afasta pro lado da continua
        return -1 if dashed_esq else 1

    def _deslocamento(self, cone, e_pista):
        """Quanto ACIMA do erro da pista o alvo precisa ir pra cumprir a folga.
        A folga e medida em larguras de cone: pede pouco com o cone longe e
        muito com ele colado. Descontar e_pista e o que faz a pista nunca
        ganhar do cone (ver docstring do modulo)."""
        folga = config.CONE_FOLGA_LARGURAS * cone.larg
        if self.lado < 0:       # cone fica a esquerda -> alvo >= x + folga
            return max(0.0, cone.cx + folga - e_pista)
        return min(0.0, cone.cx - folga - e_pista)

    def _limites(self, lane):
        """(lo, hi) em unidades de erro: quanto o alvo pode andar pra cada
        lado a partir de onde ja esta.

        Borda CONTINUA: nunca cruza. Borda PONTILHADA: pode cruzar ate
        CONE_CRUZAR_FRAC meias-larguras alem dela (ultrapassagem)."""
        m = config.CONE_OFFSET_MAX
        if lane is None or lane.hw_px is None:
            return -m, m

        metade = config.CAMERA_RES[0] / 2.0
        margem = lane.hw_px * config.CONE_MARGEM_PISTA
        esq = lane.lane_center_x - lane.hw_px + margem
        dire = lane.lane_center_x + lane.hw_px - margem

        if config.CONE_PODE_CRUZAR_PONTILHADA:
            extra = config.CONE_CRUZAR_FRAC * lane.hw_px
            s = getattr(lane, "sinal_continuo", 0.0)
            if s > 0.5:         # continua a direita -> pontilhada a esquerda
                esq -= extra
            elif s < -0.5:
                dire += extra

        # Se o alvo ja esta fora (vies de faixa), nao empurra MAIS pra fora.
        lo = min(0.0, esq - lane.center_x) / metade
        hi = max(0.0, dire - lane.center_x) / metade
        return max(lo, -m), min(hi, m)

    def _limitar_pela_pista(self, off, lane):
        lo, hi = self._limites(lane)
        return max(lo, min(hi, off))

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
