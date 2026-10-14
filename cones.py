"""
cones.py - deteccao de cones (camera frontal) e desvio.

Duas pecas, igual o resto do projeto (cada uma faz uma coisa):

  ConeVision   : frame da camera de cone -> ConeResult (onde esta o cone mais
                 perigoso, o quao perto ele esta e a que distancia da faixa
                 CONTINUA ele fica). Nao sabe nada de motor.

  ConeAvoider  : ConeResult + leitura da pista -> um DESLOCAMENTO somado ao
                 erro lateral. Nao troca o controle: o carro continua seguindo
                 a faixa pelo mesmo PID.

Como o desvio COMPOE com a pista (importante):
  NAO e erro_pista + desvio. Assim a pista, querendo ir pra esquerda, ainda
  puxava o carro pra cima de um cone que estava a esquerda. O desvio e um
  PISO: o alvo final nunca pode ficar do lado errado do cone.
      passa pela direita : alvo_final = max(erro_pista, x_cone + folga)
      passa pela esquerda: alvo_final = min(erro_pista, x_cone - folga)
  Se a pista ja quer ir pro lado seguro, o desvio e zero e nao atrapalha.

Por qual lado passa (regra do usuario):
  - cone DENTRO da faixa do carro (entre a pontilhada e a continua)
        -> cruza a pontilhada (a pontilhada pode ser cruzada);
  - cone na OUTRA faixa (do outro lado da pontilhada)
        -> se afasta, indo pro lado da continua.
  Pra saber de que lado da pontilhada o cone esta, mede-se NA MESMA IMAGEM a
  distancia do cone ate a faixa CONTINUA, em larguras de cone, na altura da
  base do cone (ConeVision._dist_solida). Na mesma imagem e na mesma altura,
  giro do carro e distancia se cancelam -- comparar posicoes entre as duas
  cameras NAO funciona: qualquer giro vira erro de dezenas de cm.

Convencao de sinais (a mesma do erro da pista):
  x do cone: -1 = borda esquerda da imagem, +1 = borda direita.
  offset > 0 = empurra o alvo pra DIREITA (carro vira a direita).
  lado -1 = passa pela DIREITA do cone | lado +1 = passa pela ESQUERDA.
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
                 "toca_base", "d_solida_w", "lado_solida")

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
        self.d_solida_w = None    # dist. do cone ate a CONTINUA (larguras de cone)
        self.lado_solida = 0      # de que lado a continua foi procurada (-1/+1)
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

    def process(self, frame, debug=False, lado_solida=0):
        """lado_solida: +1 = a faixa continua esta a DIREITA do carro,
        -1 = a ESQUERDA, 0 = nao sabe (nao mede a distancia ate ela)."""
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

            if lado_solida:
                res.lado_solida = 1 if lado_solida > 0 else -1
                res.d_solida_w = self._dist_solida(frame, melhor, res.lado_solida)

        if debug:
            res.debug_view = self._draw(frame, res)
            res.debug_mask = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        return res

    # ------------------------------------------------------------------
    @staticmethod
    def _runs_bool(row):
        idx = np.flatnonzero(row)
        if idx.size == 0:
            return []
        cortes = np.flatnonzero(np.diff(idx) > 1)
        return [(int(g[0]), int(g[-1])) for g in np.split(idx, cortes + 1)]

    def _dist_solida(self, frame, bbox, lado):
        """Distancia (em larguras de cone) do centro do cone ate a faixa
        CONTINUA, medida na altura da BASE do cone.

        Olha o chao logo ABAIXO e ao redor da base do cone (la os tracos do
        pontilhado ainda aparecem separados; mais longe eles se fundem numa
        linha so). Acha as linhas brancas finas, ajusta uma reta em cada uma
        e extrapola ate a altura exata da base -- assim a inclinacao da
        linha (carro girado) nao muda a medida. Fica com a que aparece em
        quase todas as linhas da faixa (a continua; o pontilhado tem buracos)
        do lado `lado` do cone. None se nao achou."""
        H, W = frame.shape[:2]
        x, y, w, h = bbox
        w_ref = max(float(w), h / config.CONE_ASPECTO_REAL)
        xc = x + w / 2.0

        y_base = min(H - 1, y + h)
        y0 = max(0, y_base - max(4, int(0.15 * h)))
        y1 = min(H, y_base + max(14, int(0.6 * h)) + 1)
        if y1 - y0 < 10:
            y0 = max(0, y1 - 10)
        if y1 - y0 < 6:
            return None

        band = frame[y0:y1]
        hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
        gray = cv2.GaussianBlur(cv2.cvtColor(band, cv2.COLOR_BGR2GRAY), (3, 3), 0)

        # linha = mais clara que a vizinhanca (top-hat), qualquer que seja a
        # luz. Kernel QUADRADO: com a camera baixa as linhas do chao aparecem
        # quase deitadas na imagem, e um kernel so horizontal apagaria elas.
        # O quadrado e maior que a ESPESSURA da linha (~0.25 largura de cone
        # na base; ate ~2x isso nas fileiras de baixo, mais perto do carro).
        k = int(np.clip(round(w_ref * 0.8), 7, 41)) | 1
        th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        thr = max(float(config.CONE_LINHA_TOPHAT_MIN),
                  0.4 * float(np.percentile(th, 99)))
        m = ((th >= thr) &
             (hsv[:, :, 1] <= config.SAT_MAX) &
             (hsv[:, :, 2] >= config.V_MIN_ABS))

        # tira o proprio cone (reflexos brancos do plastico)
        # (margem pequena: cone encostado na continua nao pode apagar ela)
        xa = int(max(0, x - 0.06 * w_ref))
        xb = int(min(W, x + w + 0.06 * w_ref))
        m[:, xa:xb] = False

        # segue cada linha de uma fileira pra proxima (ela e inclinada)
        n_rows = m.shape[0]
        # linha deitada na imagem = trecho horizontal LARGO por fileira
        # (espessura * sqrt(1 + inclinacao^2): ate ~2 larguras de cone).
        max_larg = max(6.0, 2.5 * w_ref)
        tol = max(6.0, 0.7 * w_ref)
        linhas = []                             # {"x": ultimo, "pts": [(r, x)]}
        for r in range(n_rows):
            ocupadas = set()
            for a, b in self._runs_bool(m[r]):
                # minusculo = ruido (e um trecho de 1 px partia a linha em
                # dois pedacos); largo demais = sapato, sombra, reflexo
                if (b - a + 1) < 4 or (b - a + 1) > max_larg:
                    continue
                c = (a + b) / 2.0
                melhor, dmin = None, tol
                for i, ln in enumerate(linhas):
                    d = abs(ln["x"] - c)
                    if d <= dmin and i not in ocupadas:
                        melhor, dmin = i, d
                borda = a <= 0 or b >= W - 1      # saiu do quadro por aqui
                if melhor is None:
                    linhas.append({"x": c, "pts": [(r, c)], "borda": borda})
                    ocupadas.add(len(linhas) - 1)
                else:
                    linhas[melhor]["x"] = c
                    linhas[melhor]["pts"].append((r, c))
                    linhas[melhor]["borda"] = borda
                    ocupadas.add(melhor)

        # x de cada linha na altura da base do cone
        r_base = y_base - y0
        lim = 0.2 * w_ref
        pres = config.CONE_LINHA_PRESENCA
        xs = []
        for ln in linhas:
            n = len(ln["pts"])
            # a linha pode sair do quadro pela lateral nas fileiras de baixo:
            # nesse caso a presenca e contada so ate onde ela ainda aparece
            rows_ok = (ln["pts"][-1][0] + 1) if ln["borda"] else n_rows
            if n < 5 or n / max(rows_ok, 1) < pres:
                continue
            rr = np.array([p[0] for p in ln["pts"]], dtype=float)
            cc = np.array([p[1] for p in ln["pts"]], dtype=float)
            if rr.max() - rr.min() >= 4:
                a_, b_ = np.polyfit(rr, cc, 1)
                x_b = a_ * r_base + b_ if abs(a_) <= 3.0 else float(cc.mean())
            else:
                x_b = float(cc.mean())
            xs.append(x_b)

        if lado > 0:
            cands = [v for v in xs if v > xc + lim]
        else:
            cands = [v for v in xs if v < xc - lim]
        if not cands:
            return None
        return abs(min(cands, key=lambda v: abs(v - xc)) - xc) / w_ref

    @staticmethod
    def _draw(frame, res):
        vis = frame.copy()
        H, W = vis.shape[:2]
        cv2.line(vis, (W // 2, 0), (W // 2, H), (255, 255, 255), 1)
        if res.found:
            x, y, w, h = res.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            txt = f"x {res.cx:+.2f} prox {res.proximidade:.2f}"
            if res.d_solida_w is not None:
                txt += f" cont {res.d_solida_w:.1f}w"
            cv2.putText(vis, txt, (x, max(12, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        return vis


# ======================================================================
# DESVIO
# ======================================================================
class ConeAvoider:
    def __init__(self):
        self.offset = 0.0          # saida suavizada (soma no erro da pista)
        self.lado = 0              # -1 passa pela direita, +1 pela esquerda
        self.prox = 0.0            # proximidade do cone em uso (p/ freio e HUD)
        self.ativo = False
        self.parar = False         # cone colado na frente: Main zera a velocidade
        self.na_faixa = None       # True = cone na faixa do carro, False = outra
        self.dist_cont_cm = None   # distancia do cone ate a continua (cm)

        self._alvo = 0.0           # ultimo deslocamento "bruto" com cone visivel
        self._t_visto = None       # quando viu cone pela ultima vez
        self._prox_visto = 0.0
        self._alt_ant = None
        self._disc_t = None        # desde quando a proposta discorda do lado
        self._t_ant = None         # p/ limitar a velocidade de variacao

    def reset(self):
        self.offset = 0.0
        self.lado = 0
        self.prox = 0.0
        self.ativo = False
        self.parar = False
        self.na_faixa = None
        self.dist_cont_cm = None
        self._alvo = 0.0
        self._t_visto = None
        self._prox_visto = 0.0
        self._alt_ant = None
        self._disc_t = None
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

            # rede de seguranca (desligada por padrao): cone colado BEM na
            # frente -> para
            if (config.CONE_PARAR_ALTURA > 0.0 and
                    abs(cone.cx) < config.CONE_PARAR_X and
                    (cone.altura >= config.CONE_PARAR_ALTURA or cone.toca_base)):
                self.parar = True
            self._alvo = alvo
            self._t_visto = agora
            self._prox_visto = cone.proximidade
            self._alt_ant = cone.altura
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

    def _propor_lado(self, cone, lane, e_pista, sobra):
        """Lado que a regra manda, frame a frame (sem travar)."""
        self.na_faixa = None
        self.dist_cont_cm = None

        # 1) regra do usuario, medida NA IMAGEM DO CONE
        if cone.d_solida_w is not None and cone.lado_solida != 0:
            d = cone.d_solida_w
            larg_pista = config.PISTA_LARGURA_CM / config.CONE_LARGURA_CM
            na_linha = config.CONE_NA_LINHA_CM / config.CONE_LARGURA_CM
            continua_dir = cone.lado_solida > 0
            self.dist_cont_cm = d * config.CONE_LARGURA_CM

            if abs(d - larg_pista) < na_linha:
                # em cima da pontilhada: vai pro lado com mais espaco
                return -1 if sobra[-1] >= sobra[+1] else 1
            self.na_faixa = d < larg_pista
            if self.na_faixa:
                # cruza a pontilhada: esquerda se a continua e a direita
                return 1 if continua_dir else -1
            # na outra faixa: afasta, pro lado da continua
            return -1 if continua_dir else 1

        # 2) sem a continua na imagem: o lado que a PROPRIA pista ja quer
        # (alvo da pista a esquerda do cone -> passa pela esquerda). Compara
        # so direcoes (bearing), entao nao sofre com giro do carro.
        diff = e_pista - cone.cx
        if abs(diff) >= config.CONE_ZONA_CENTRO:
            return 1 if diff < 0 else -1

        # 3) empate: lado da pontilhada (e onde pode cruzar)
        s = getattr(lane, "sinal_continuo", 0.0) if lane is not None else 0.0
        if s > 0.5:
            return 1
        if s < -0.5:
            return -1
        return -1 if config.CONE_LADO_PADRAO == "direita" else 1

    def _decidir_lado(self, cone, agora, e_pista, lane):
        """Mantem o lado ESTAVEL enquanto for o mesmo cone.

        Uma proposta que discorda do lado atual so vence se persistir por
        CONE_TROCA_PERSISTE_S (a medida e ruidosa e o carro balanca), e so
        com o cone ainda longe -- trocar de lado colado e pior que seguir."""
        sobra = self._sobras(cone, e_pista, lane)
        prop = self._propor_lado(cone, lane, e_pista, sobra)

        # a regra manda um lado onde nao cabe, mas o outro cabe: usa o outro
        if sobra[prop] < -0.05 and sobra[-prop] >= 0.0:
            prop = -prop

        # e OUTRO cone? sumiu por um tempo, ou o mais alto mudou (o cone
        # que o carro acabou de passar saiu e apareceu um mais longe)
        outro = (self.lado == 0 or self._t_visto is None or
                 agora - self._t_visto > config.CONE_RECONHECE_S or
                 (self._alt_ant is not None and
                  cone.altura < 0.65 * self._alt_ant))
        if outro:
            self.lado = prop
            self._disc_t = None
            return

        if prop == self.lado:
            self._disc_t = None
            return

        if self._disc_t is None:
            self._disc_t = agora
        if (agora - self._disc_t >= config.CONE_TROCA_PERSISTE_S and
                cone.altura < config.CONE_TROCA_LADO_ALTURA_MAX):
            self.lado = prop
            self._disc_t = None

    def _deslocamento(self, cone, e_pista):
        """Quanto ACIMA do erro da pista o alvo precisa ir pra cumprir a folga.
        A folga e medida em larguras de cone: pede pouco com o cone longe e
        muito com ele colado. Descontar e_pista e o que faz a pista nunca
        ganhar do cone (ver docstring do modulo)."""
        folga = config.CONE_FOLGA_LARGURAS * cone.larg
        if self.lado < 0:       # passa pela direita -> alvo >= x + folga
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
            self._alt_ant = None
            self._disc_t = None
            self.lado = 0
            self.prox = 0.0
            self.ativo = False
            return 0.0
        return self._alvo * f
