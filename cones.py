"""
cones.py - deteccao de cones (camera frontal) e desvio.

Duas pecas:

  ConeVision   : frame da camera de cone -> ConeResult (onde esta o cone mais
                 perigoso, o quao perto ele esta e a que distancia da faixa
                 CONTINUA ele fica). Nao sabe nada de motor.

  ConeAvoider  : ConeResult + leitura da pista -> um DESLOCAMENTO do alvo da
                 pista. O carro continua seguindo a faixa pelo mesmo PID; o
                 cone so diz "onde, na faixa, o alvo deve ficar".

COMO O DESVIO FUNCIONA (lei de POSICAO lateral, em cm da faixa)
  1. Mede-se onde o cone esta na faixa: distancia dele ate a faixa CONTINUA
     (ConeVision._dist_solida), na imagem do cone, na altura da base dele.
     Como o cone e FIXO, essa posicao (cm a partir do centro da faixa) e a
     mesma o tempo todo; usa-se a mediana das medidas feitas de longe.
  2. Regra do usuario: cone DENTRO da faixa do carro -> cruza a pontilhada;
     cone na OUTRA faixa -> se afasta, pro lado da continua.
  3. Alvo lateral = posicao do cone -/+ CONE_FOLGA_CM (meia largura do carro
     + meia do cone + margem), limitado pelas bordas (nao passa da continua e
     nao vai longe demais pra outra faixa).
  4. O deslocamento (alvo desejado - onde o alvo da pista esta hoje) e somado
     ao ERRO da pista e passa pelo PID; e um PISO: so empurra pra longe do
     cone. Depois que o cone passa, o deslocamento volta DEVAGAR a zero.

POR QUE NAO "MIRAR O CONE" (pure pursuit) E NAO COMPARAR AS 2 CAMERAS
  - mirar um ponto ao lado do cone gira o carro cada vez mais forte e nao
    para quando o cone passa;
  - comparar posicoes entre a camera de pista e a de cone oscila com o giro
    do carro (erro de dezenas de cm); a medida na MESMA imagem nao.

DETALHES QUE IMPORTAM (cada um foi uma batida no simulador, ver sim/)
  - a continua e a linha mais EXTERNA daquele lado (a pontilhada, de perto,
    parece continua); a regua e a largura da faixa NA MESMA FILEIRA (em curva
    a pista fica girada e a distancia horizontal e inflada por 1/cos);
  - cone novo nao derruba o desvio de um cone que o carro ainda esta passando;
  - o lado so troca se a medida discordar por um tempo, com o cone longe.

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
                 "toca_base", "d_solida_w", "lado_solida", "dx_pont_w")

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
        self.dx_pont_w = None     # cone - PONTILHADA na mesma fileira, em larguras de
                                  # cone (+ = cone a DIREITA dela). Imune ao giro do
                                  # carro e nao precisa ver a continua.
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

            self._dx_pont = None
            if lado_solida:
                res.lado_solida = 1 if lado_solida > 0 else -1
            res.d_solida_w = self._dist_solida(frame, melhor, res.lado_solida)
            res.dx_pont_w = self._dx_pont

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
        pres = config.CONE_LINHA_PRESENCA
        todas = []                              # (x_na_base, presenca)
        for ln in linhas:
            n = len(ln["pts"])
            # a linha pode sair do quadro pela lateral nas fileiras de baixo:
            # nesse caso a presenca e contada so ate onde ela ainda aparece
            rows_ok = (ln["pts"][-1][0] + 1) if ln["borda"] else n_rows
            if n < 3:
                continue
            rr = np.array([p[0] for p in ln["pts"]], dtype=float)
            cc = np.array([p[1] for p in ln["pts"]], dtype=float)
            if rr.max() - rr.min() >= 4:
                a_, b_ = np.polyfit(rr, cc, 1)
                # com a camera baixa as linhas ficam quase deitadas (3-8 px de
                # x por fileira): limite alto, senao a media das fileiras
                # (longe da base) entrava no lugar da extrapolacao
                x_b = a_ * r_base + b_ if abs(a_) <= 12.0 else float(cc.mean())
            else:
                x_b = float(cc.mean())
            todas.append((x_b, n / max(rows_ok, 1), n))

        # PONTILHADA (linha de presenca parcial: tem buracos) mais proxima do
        # cone, na MESMA fileira da base: em larguras de cone. De perto (carro
        # parado ~1 m) os tracos aparecem separados; a posicao relativa nao
        # muda com o giro do carro e nao depende de a continua estar no quadro.
        pont = [(x, p, n) for (x, p, n) in todas
                if 0.12 <= p <= 0.75 and n >= 3 and abs(x - xc) > 0.15 * w_ref
                and abs(x - xc) / w_ref <= 2.5 * config.PISTA_LARGURA_CM / config.CONE_LARGURA_CM]
        if pont:
            x_p = min(pont, key=lambda t: abs(t[0] - xc))[0]
            self._dx_pont = (xc - x_p) / w_ref
        if lado == 0:
            return None

        lim = 0.2 * w_ref
        sinal = 1.0 if lado > 0 else -1.0
        # so conta linha a uma distancia plausivel: a continua fica, no
        # maximo, a ~2 larguras de faixa do cone (cone na outra faixa).
        d_max = 2.5 * config.PISTA_LARGURA_CM / config.CONE_LARGURA_CM
        solidas = [x for (x, p, n) in todas
                   if p >= pres and n >= 5 and (x - xc) * sinal > lim
                   and abs(x - xc) / w_ref <= d_max * 2.0]
        if not solidas:
            return None
        # a CONTINUA e a mais EXTERNA daquele lado. A pontilhada, de perto,
        # tem tracos que se fundem e tambem passa por "presente em todas as
        # fileiras"; pegar a mais proxima do cone confundia as duas.
        x_s = max(solidas, key=lambda x: (x - xc) * sinal)
        d_cone = abs(x_s - xc)

        # REGUA = a propria largura da faixa NESSA fileira (pontilhada ate a
        # continua), nao a largura do cone: em curva a pista fica girada em
        # relacao a camera e uma linha horizontal da imagem cruza as linhas
        # esticada por 1/cos(giro) -- as duas distancias sao inflacionadas
        # igual, entao a RAZAO nao muda. Pontilhada = a linha mais proxima da
        # continua do lado de dentro da faixa (presenca parcial: tem buracos).
        larg_min = 0.9 * w_ref * (config.PISTA_LARGURA_CM / config.CONE_LARGURA_CM) * 0.5
        exp_px = w_ref * (config.PISTA_LARGURA_CM / config.CONE_LARGURA_CM)
        opcoes = []                 # (desvio_de_plausibilidade, largura_da_faixa_px)
        for (x, p, n) in todas:
            if (x_s - x) * sinal <= larg_min or n < 3:
                continue
            S = abs(x_s - x)
            if p <= 0.9:
                # tem buracos: e a PONTILHADA, a 1 faixa da continua
                if p >= 0.15 and 0.8 <= S / exp_px <= 2.6:
                    opcoes.append((abs(S / exp_px - 1.0) - 0.05, S))
            else:
                # linha continua do lado de dentro: ou e a pontilhada fundida
                # (1 faixa) ou a linha EXTERNA do outro lado (2 faixas). Pega a
                # interpretacao em que a largura da faixa fica mais proxima da
                # esperada pela largura do cone (kappa ~ 1, ate ~2.6 em curva).
                if 0.8 <= S / exp_px <= 2.6:
                    opcoes.append((abs(S / exp_px - 1.0), S))
                if 0.8 <= S / (2.0 * exp_px) <= 2.6:
                    opcoes.append((abs(S / (2.0 * exp_px) - 1.0), S / 2.0))
        if opcoes:
            larg_faixa_px = min(opcoes)[1]
            if larg_faixa_px > 1.0:
                frac = d_cone / larg_faixa_px      # 0 = na continua, 1 = na pontilhada
                return frac * (config.PISTA_LARGURA_CM / config.CONE_LARGURA_CM)
        return d_cone / w_ref

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
        self.liberado = True       # manobra.py: False = so OBSERVA o cone (nao desvia ainda)
        self.na_faixa = None       # True = cone na faixa do carro, False = outra
        self.na_outra = False      # o CARRO esta na faixa do outro lado da pontilhada
        self.faixa_outra = False   # o carro FICA na outra faixa (nao volta pra de antes ao passar o cone)
        self._faixa_alvo_outra = False
        self.off_base = 0.0        # deslocamento que mantem o carro na faixa em que esta
        self._alt_pico = 0.0       # maior altura do cone que esta sendo passado (distingue cone NOVO)
        self._congelado = False    # decisao (lado + posicao do cone) travada durante o desvio
        self._w_fixo = None
        self._t_outra = None       # desde quando o alvo planejado esta na outra faixa
        self.dist_cont_cm = None   # distancia do cone ate a continua (cm)
        self.larg = 0.0            # largura do cone (x norm.) -> ganho do esterco
        self._lado_medido = False  # o lado atual veio de MEDIDA (nao de palpite)
        self.modo_pos = False      # True = lei de POSICAO lateral (cm na faixa)
        self.w_alvo_cm = None      # onde o alvo do carro deve ficar (cm do centro)
        self.w_cone_cm = None      # onde o cone esta (cm do centro da pista)
        self._w_mem = None         # posicao MEDIDA deste cone (cm da pista)
        self._w_obs = []           # medidas recentes (pra mediana)
        self._w_geo = []           # estimativas geometricas (carro + lat), mediana
        self._w_lane = None        # medida AGORA pela camera da pista (cm)
        self._w_lane_obs = []      # medidas recentes pela camera da pista
        self.conf = 1.0            # 1 = posicao medida, <1 = so estimativa

        self._alvo = 0.0           # ultimo deslocamento "bruto" com cone visivel
        self._t_visto = None       # quando viu cone pela ultima vez
        self._prox_visto = 0.0
        self._alt_ant = None
        self._disc_t = None        # desde quando a proposta discorda do lado
        self._t_ant = None         # p/ limitar a velocidade de variacao

    def reset(self, manter_faixa=False):
        if not manter_faixa:
            self.offset = 0.0
            self.faixa_outra = False
            self._faixa_alvo_outra = False
        self.lado = 0
        self._alt_pico = 0.0
        self._congelado = False
        self._w_fixo = None
        self.prox = 0.0
        self.ativo = False
        self.parar = False
        self.na_faixa = None
        self.dist_cont_cm = None
        self.larg = 0.0
        self.na_outra = False
        self._t_outra = None
        self._lado_medido = False
        self.modo_pos = False
        self.w_alvo_cm = None
        self.w_cone_cm = None
        self._w_mem = None
        self._w_obs = []
        self._w_geo = []
        self._w_lane = None
        self._w_lane_obs = []
        self.conf = 1.0
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

        # COMPROMISSO: o carro ainda esta passando um cone (chegou a ficar
        # perto e o desvio dele ainda esta no periodo de "segurar"). Um cone
        # NOVO e mais longe nao pode derrubar esse desvio agora: o carro
        # soltaria o lado, voltaria pro centro e bateria no cone que ainda
        # esta ao lado dele (a traseira ainda nao passou).
        if (visivel and self._t_visto is not None and self._alt_ant is not None
                and (agora - self._t_visto) < config.CONE_SEGURAR_S
                and abs(self._alvo) > 0.08
                and cone.altura < 0.8 * self._alt_ant
                and cone.altura < config.CONE_COMPROMISSO_LONGE):
            visivel = False

        # com a manobra (para -> analisa -> desvia), um cone BEM MENOR que o que
        # esta sendo passado e OUTRO cone, mais longe: nao age nele agora (a
        # manobra para por ele quando chegar no gatilho); segue segurando o
        # desvio atual.
        if (visivel and self.liberado and config.CONE_PARA_E_ANALISA
                and self._alt_pico > 0.3 and cone.altura < 0.5 * self._alt_pico):
            visivel = False

        self.parar = False
        self._w_lane = None
        if visivel and lane is not None and config.CONE_NA_PISTA:
            w_l = getattr(lane, "cone_w_cm", None)
            # a camera da pista so enxerga o cone mais PROXIMO; so associa
            # quando a camera frontal tambem o ve perto (mesmo cone)
            if w_l is not None and cone.altura >= config.CONE_PISTA_ASSOCIA_ALTURA:
                self._w_lane = w_l
        if visivel:
            e_pista = lane.error if lane is not None else 0.0
            base_off = self._off_base(lane)
            self.off_base = base_off
            self._decidir_lado(cone, agora, e_pista, lane)
            # a folga ja e geometrica (larguras de cone), entao o pedido
            # cresce sozinho com a aproximacao; a rampa so evita reagir a
            # ruido muito longe.
            rampa = 1.0 if cone.proximidade > 0.0 else 0.0
            off_pos = self._offset_posicao(cone, lane)
            if off_pos is not None:
                # lei de POSICAO lateral (cm na faixa): o seguidor de faixa
                # leva o carro suavemente ate la e volta sozinho depois
                self.modo_pos = True
                alvo = off_pos * rampa * config.CONE_GANHO * self.conf
                alvo = max(-config.CONE_OFFSET_MAX, min(config.CONE_OFFSET_MAX, alvo))
            else:
                # reserva (sem pista/contínua conhecida): mira um ponto ao
                # lado do cone, em direcao (pure pursuit)
                self.modo_pos = False
                alvo = self._deslocamento(cone, e_pista) * rampa * config.CONE_GANHO
                alvo = self._limitar_pela_pista(alvo, lane)

            # CONE NO HORIZONTE: mede e decide, mas so AGE quando chegar perto
            # (ou se ja esta no meio de uma manobra). Agir de longe era a
            # parte "imprecisa": o carro se deslocava antes de saber direito.
            medido = bool(self._w_lane_obs) or self._w_mem is not None
            alt_min = (config.CONE_ACAO_ALTURA if medido
                       else config.CONE_ACAO_ALTURA_SEM_MEDIDA)
            agindo = (cone.altura >= alt_min or
                      abs(self.offset) > 0.1 or abs(self._alvo) > 0.1)
            if config.CONE_PARA_E_ANALISA:
                # quem manda e a manobra (para -> analisa -> desvia): so
                # desvia quando liberado; antes disso so observa
                agindo = self.liberado
            if not agindo:
                alvo = base_off
            if not agindo:
                alvo = base_off

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
            self.larg = cone.larg
            self.ativo = agindo
            if agindo and self.liberado:
                self._alt_pico = max(self._alt_pico, cone.altura)
            if agindo and self.w_alvo_cm is not None:
                self.faixa_outra = self._faixa_alvo_outra   # fica na faixa pra onde foi
            self._atualiza_faixa(agindo, agora)
        else:
            alvo = self._sem_cone(agora, lane)

        # suaviza (EMA) e ainda limita a velocidade de variacao: o cone
        # aparece/some de repente e isso vira degrau no volante.
        dt = 1 / 30.0 if self._t_ant is None else max(0.0, min(0.2, agora - self._t_ant))
        self._t_ant = agora
        b = config.CONE_OFFSET_SUAVIZACAO
        novo = b * self.offset + (1.0 - b) * alvo
        # indo: pode variar rapido (o carro precisa sair a tempo). VOLTANDO
        # (modulo diminuindo): bem mais devagar -- o seguidor de faixa, feito
        # pra erros pequenos, nao acompanha um alvo que "solta" 30 cm de uma
        # vez e passa direto pela faixa (cruzava ate a continua).
        voltando = abs(novo) < abs(self.offset)
        if voltando:
            taxa = config.CONE_RETORNO_SLEW
        elif config.CONE_PARA_E_ANALISA and self.liberado:
            taxa = config.CONE_OFFSET_SLEW_DESVIO   # saindo do repouso: ja sabe o lado, vai direto
        else:
            taxa = config.CONE_OFFSET_SLEW
        passo = taxa * dt
        self.offset += max(-passo, min(passo, novo - self.offset))
        if abs(self.offset) < 0.005 and not self.ativo:
            self.offset = 0.0
        return self.offset

    def no_caminho(self, cone, lane):
        """O cone esta onde o carro pode bater? False = bem fora da pista (alem
        da linha externa de qualquer lado): nao vale parar por ele. Sem
        estimativa confiavel devolve True (na duvida, trata como obstaculo)."""
        if cone is None or not cone.found:
            return True
        lw = config.PISTA_LARGURA_CM
        s = getattr(lane, "sinal_continuo", 0.0) if lane is not None else 0.0
        if abs(s) < 0.5:
            return True
        w = self._w_pont(cone, lane)
        if w is None and cone.d_solida_w is not None and cone.lado_solida != 0:
            d = cone.d_solida_w * config.CONE_LARGURA_CM
            w = (lw / 2.0 - d) if cone.lado_solida > 0 else (-lw / 2.0 + d)
        if w is None:
            w = self._w_geometrico(cone, lane, grava=False)
        if w is None:
            return True
        # pista = da linha externa esquerda (-1.5*lw) ate a continua (+0.5*lw)
        return (-1.5 * lw - config.CONE_FORA_PISTA_CM) <= w <= (0.5 * lw + config.CONE_FORA_PISTA_CM)

    def congelar_decisao(self, cone, lane):
        """Trava o lado e a posicao do cone na que foi medida COM O CARRO
        PARADO. Com o cone ao lado do carro a imagem nao serve mais pra medir
        (a posicao virava -25, -56, +2 cm e o lado trocava no meio da
        passagem)."""
        w, _ = self._w_cone(cone, lane)
        if w is not None and self.lado != 0:
            self._w_fixo = w
            self._congelado = True

    def descongelar(self):
        self._congelado = False
        self._w_fixo = None

    def reiniciar_medidas(self):
        self._congelado = False
        self._w_fixo = None
        """Esquece as medidas da aproximacao (de longe, com o carro andando e
        balancando) pra medir de novo com o carro parado."""
        self._w_mem = None
        self._w_obs = []
        self._w_geo = []
        self._w_lane = None
        self._w_lane_obs = []
        self._lado_medido = False
        self.lado = 0
        self._disc_t = None
        self._alt_pico = 0.0

    def cone_do_lado(self, cone):
        """True se ha um cone A VISTA, perto, do lado que esta sendo passado
        (lado +1 = passa pela esquerda -> o cone fica a direita). So nesse caso
        e proibido esterçar pra esse lado: depois que o cone sai de vista o
        carro precisa poder voltar pra faixa."""
        if (cone is None or not cone.found or not self.ativo or self.lado == 0
                or cone.altura < config.CONE_TRAVA_ALTURA_MIN):
            return False
        return cone.cx > -0.15 if self.lado > 0 else cone.cx < 0.15

    def ganho_esterco(self, kp):
        """Ganho aplicado ao deslocamento (so no modo 'esterco extra')."""
        if self.modo_pos:
            return kp * config.CONE_GANHO_POS
        return max(kp, config.CONE_ESTERCO_GANHO * self.larg)

    def _atualiza_faixa(self, agindo, agora):
        """Mantem 'o carro esta na outra faixa?' a partir do que o proprio
        desvio comanda: alvo planejado do outro lado da pontilhada por um
        tempo (da tempo do carro chegar la) -> esta na outra; alvo planejado
        na propria faixa -> voltou."""
        if self.w_alvo_cm is None or not agindo:
            return
        lw = config.PISTA_LARGURA_CM
        # distancia do alvo planejado ate a pontilhada, + = pro lado da outra
        # faixa. A pontilhada fica a lw/2 do centro da sua faixa.
        if getattr(self, '_s_cont', 1.0) >= 0:      # continua a direita
            alvo_na_outra = self.w_alvo_cm < -lw / 2.0 - 2.0
        else:                                       # continua a esquerda
            alvo_na_outra = self.w_alvo_cm > lw / 2.0 + 2.0
        if alvo_na_outra == self.na_outra:
            self._t_outra = None
            return
        if self._t_outra is None:
            self._t_outra = agora
        if agora - self._t_outra >= config.CONE_FAIXA_TEMPO_S:
            self.na_outra = alvo_na_outra
            self._t_outra = None

    def fator_velocidade(self):
        """Multiplicador (<=1) da velocidade: desvia mais devagar."""
        if not self.ativo:
            return 1.0
        # so freia se de fato esta desviando (cone fora do caminho nao freia)
        relevancia = min(1.0, abs(self.offset) / 0.15)
        return 1.0 - config.CONE_FREIO_MAX * self.prox * relevancia

    # ------------------------------------------------------------------
    def _lado_da_pontilhada(self, lane):
        """Lado (-1 passa pela direita / +1 pela esquerda) que CRUZA a
        pontilhada. None se nao se sabe qual lado e a pontilhada."""
        s = getattr(lane, "sinal_continuo", 0.0) if lane is not None else 0.0
        if s > 0.5:
            return 1            # continua a direita -> pontilhada a esquerda
        if s < -0.5:
            return -1
        return None

    def _propor_lado(self, cone, lane):
        """-> (lado, medido). `medido` = veio da distancia ate a continua na
        imagem do cone; senao e so um palpite inicial."""
        self.na_faixa = None
        self.dist_cont_cm = None
        pont = self._lado_da_pontilhada(lane)

        # 0) posicao do cone medida PELA CAMERA DA PISTA (a melhor medida:
        # base do cone x centro da faixa na mesma fileira da imagem)
        lista = self._w_lane_obs + ([self._w_lane] if self._w_lane is not None else [])
        if lista and pont is not None:
            w = float(np.median(lista[-7:]))
            lw = config.PISTA_LARGURA_CM
            dentro = w if pont > 0 else -w          # + = rumo a continua
            cone_na_outra = dentro < -lw / 2.0 - config.CONE_NA_LINHA_CM
            self.na_faixa = not cone_na_outra
            return (-pont if cone_na_outra else pont), True

        # 0) cone x PONTILHADA na mesma fileira (a regra do usuario, literal)
        wp = self._w_pont_ok(cone, lane)
        if wp is not None and pont is not None:
            lw = config.PISTA_LARGURA_CM
            dentro = wp if pont > 0 else -wp
            cone_na_outra = dentro < -lw / 2.0 - config.CONE_NA_LINHA_CM
            self.na_faixa = not cone_na_outra
            return (-pont if cone_na_outra else pont), True

        # 1) a regra do usuario, medida NA IMAGEM DO CONE
        if (cone.d_solida_w is not None and cone.lado_solida != 0 and
                cone.altura >= config.CONE_MEDIDA_ALTURA_MIN and
                (cone.altura <= config.CONE_MEDIDA_ALTURA_MAX or self._w_mem is None)):
            d = cone.d_solida_w
            larg_pista = config.PISTA_LARGURA_CM / config.CONE_LARGURA_CM
            na_linha = config.CONE_NA_LINHA_CM / config.CONE_LARGURA_CM
            continua_dir = cone.lado_solida > 0
            cruza = 1 if continua_dir else -1       # lado que cruza a pontilhada
            self.dist_cont_cm = d * config.CONE_LARGURA_CM
            if abs(d - larg_pista) < na_linha:
                # em cima da pontilhada: so cabe passando pelo lado de FORA
                # da faixa (entre o cone e a continua nao cabe o carro)
                return cruza, True
            self.na_faixa = d < larg_pista
            if self.na_faixa:
                return cruza, True              # cone na faixa: cruza a pontilhada
            return -cruza, True                 # cone na outra: afasta

        # 2) a continua nao esta na imagem agora, mas este cone ja foi medido
        if self._w_mem is not None and pont is not None:
            w = self._w_mem
            larg_pista = config.PISTA_LARGURA_CM
            continua_dir = pont > 0          # pont>0 -> pontilhada a esquerda
            # coordenada "para dentro da faixa": + = em direcao a continua
            dentro = w if continua_dir else -w
            if dentro > -larg_pista / 2.0 - config.CONE_NA_LINHA_CM:
                return pont, True            # na sua faixa (ou na linha): cruza
            return -pont, True               # na outra faixa: afasta

        # 3) sem medida da posicao do cone: usa EM QUAL FAIXA O CARRO ESTA.
        # Um cone de frente (x perto de 0) esta no caminho do carro, ou seja,
        # NA FAIXA DELE. Um cone bem a direita/esquerda do carro esta na
        # faixa daquele lado da pontilhada. A faixa do carro (propria ou a
        # outra, do outro lado da pontilhada) vem de self.na_outra, que o
        # proprio desvio mantem (cruzou -> esta na outra; voltou -> propria).
        if pont is not None and config.CONE_USA_POS_CARRO:
            # onde o carro REALMENTE esta (visao da pista) + o quanto o cone
            # esta ao lado dele (larguras de cone). Nao precisa ver linha
            # nenhuma e nao depende de maquina de estados.
            w = self._w_geometrico(cone, lane, grava=False)
            if w is not None:
                lw = config.PISTA_LARGURA_CM
                dentro = w if pont > 0 else -w        # + = rumo a continua
                cone_na_outra = dentro < -lw / 2.0 - config.CONE_NA_LINHA_CM
                self.na_faixa = not cone_na_outra
                return (-pont if cone_na_outra else pont), False
        if pont is not None and not config.CONE_USA_FAIXA_CARRO:
            # versao anterior (sem saber em qual faixa o carro esta): se o carro
            # acabou de se deslocar, cone de frente passa pelo lado de volta;
            # senao, cruza a pontilhada.
            if (config.CONE_PALPITE_VOLTA and self.lado != 0 and abs(self._alvo) > 0.2
                    and self._t_visto is not None):
                if self.lado == pont and cone.cx < 0.3:
                    return -pont, False
                return self.lado, False
            return pont, False
        if pont is not None:
            # posicao lateral estimada do cone (cm do centro da SUA faixa,
            # + = direita): onde o carro esta (centro da propria faixa, ou
            # da outra se ele cruzou) + o quanto o cone esta ao lado dele,
            # medido em larguras de cone (nao depende do angulo nem da
            # distancia).
            lw, cw = config.PISTA_LARGURA_CM, config.CONE_LARGURA_CM
            w_carro = 0.0
            if self.na_outra:
                w_carro = -lw if pont > 0 else lw      # centro da faixa do outro lado
            w_geo = w_carro + cone.lat * cw
            if pont > 0:                               # pontilhada a esquerda
                cone_na_outra = w_geo < -lw / 2.0
            else:
                cone_na_outra = w_geo > lw / 2.0
            self.na_faixa = not cone_na_outra
            return (-pont if cone_na_outra else pont), False
        return (-1 if config.CONE_LADO_PADRAO == "direita" else 1), False

    def _decidir_lado(self, cone, agora, e_pista, lane):
        """Mantem o lado ESTAVEL enquanto for o mesmo cone.

        Medida nova que discorda do lado atual so vence se persistir por
        CONE_TROCA_PERSISTE_S, e so com o cone ainda longe -- trocar de lado
        colado e pior que seguir. Sem medida, nunca troca (so o palpite
        inicial)."""
        if self._congelado:
            return              # decidido com o carro parado: nao muda no meio da passagem
        prop, medido = self._propor_lado(cone, lane)

        # e OUTRO cone? sumiu por um tempo, ou o mais alto mudou (o cone
        # que o carro acabou de passar saiu e apareceu um mais longe)
        outro = (self.lado == 0 or self._t_visto is None or
                 agora - self._t_visto > config.CONE_RECONHECE_S or
                 (self._alt_ant is not None and
                  cone.altura < 0.65 * self._alt_ant))
        if outro:
            self.lado = prop
            self._lado_medido = medido
            self._disc_t = None
            self._w_mem = None      # a posicao medida era do cone anterior
            self._w_obs = []
            self._w_geo = []
            self._w_lane_obs = []
            return

        # o lado atual era so um PALPITE (a faixa continua ainda nao tinha
        # sido identificada): a primeira medida real o substitui na hora,
        # mesmo com o cone ja perto -- o palpite errado e pior que trocar.
        if medido and not self._lado_medido:
            # o lado atual era so palpite: a primeira medida real o substitui
            self.lado = prop
            self._lado_medido = True
            self._disc_t = None
            return
        if (not medido and not self._lado_medido and abs(self._alvo) < 0.05
                and abs(self.offset) < 0.05):
            # ainda nao comecou a desviar: a estimativa melhora a cada frame
            self.lado = prop
            self._disc_t = None
            return
        if not medido or prop == self.lado:
            self._disc_t = None
            return

        if self._disc_t is None:
            self._disc_t = agora
        # medida da camera da pista e MELHOR quanto mais perto (leitura de
        # longe e ruidosa e foi o que travou o lado errado): vale trocar
        # mesmo com o cone perto
        da_pista = bool(self._w_lane_obs) or self._w_lane is not None
        if (agora - self._disc_t >= config.CONE_TROCA_PERSISTE_S and
                (da_pista or cone.altura < config.CONE_TROCA_LADO_ALTURA_MAX)):
            self.lado = prop
            self._lado_medido = True
            self._disc_t = None

    def _w_pont(self, cone, lane):
        """Posicao do cone (cm do centro da faixa direita, + = direita) pela
        distancia ate a PONTILHADA na imagem do cone. None se nao deu."""
        s = getattr(lane, "sinal_continuo", 0.0) if lane is not None else 0.0
        if cone.dx_pont_w is None or abs(s) < 0.5:
            return None
        lw, cw = config.PISTA_LARGURA_CM, config.CONE_LARGURA_CM
        dx = cone.dx_pont_w * cw
        if abs(dx) > 1.2 * lw:
            return None
        w_pont = -lw / 2.0 if s > 0 else lw / 2.0     # onde esta a pontilhada
        return w_pont + dx

    def _w_pont_ok(self, cone, lane):
        """_w_pont, mas descarta se discordar muito da geometria (carro +
        lat): a linha mais proxima do cone pode nao ser a pontilhada."""
        wp = self._w_pont(cone, lane)
        if wp is None:
            return None
        g = self._w_geometrico(cone, lane, grava=False)
        if g is not None and abs(wp - g) > config.CONE_PONT_TOL_CM:
            return None
        return wp

    def _w_cone(self, cone, lane):
        if self._congelado and self._w_fixo is not None:
            self.conf = 1.0
            return self._w_fixo, True
        return self._w_cone_medir(cone, lane)

    def _w_cone_medir(self, cone, lane):
        """Posicao lateral do cone, em cm a partir do CENTRO da pista
        (+ = direita). (valor, medido). medido=True vem da distancia ate a
        continua na imagem do cone (exata, nao depende do giro do carro);
        senao e uma estimativa pela geometria (aceita a longe)."""
        lw, cw = config.PISTA_LARGURA_CM, config.CONE_LARGURA_CM
        g = self._w_geometrico(cone, lane, grava=True)     # mantem a memoria geometrica
        wp = self._w_pont_ok(cone, lane)
        if wp is not None:
            if cone.altura >= config.CONE_MEDIDA_ALTURA_MIN:
                self._w_obs.append(wp)
                if len(self._w_obs) > 25:
                    self._w_obs.pop(0)
                self._w_mem = float(np.median(self._w_obs))
            self.conf = 1.0
            return self._w_mem if self._w_mem is not None else wp, True
        if self._w_lane is not None:
            self._w_lane_obs.append(self._w_lane)
            if len(self._w_lane_obs) > 12:
                self._w_lane_obs.pop(0)
        if self._w_lane_obs:
            # o cone e fixo: mediana das ultimas medidas da camera da pista
            w = float(np.median(self._w_lane_obs[-7:]))
            self._w_mem = w
            self.conf = 1.0
            return w, True
        if (cone.d_solida_w is not None and cone.lado_solida != 0 and
                cone.altura >= config.CONE_MEDIDA_ALTURA_MIN):
            d = cone.d_solida_w * cw
            w = (lw / 2.0 - d) if cone.lado_solida > 0 else (-lw / 2.0 + d)
            # o cone e FIXO na pista: a posicao dele em cm na faixa nao muda
            # quando o carro se mexe. A medida e precisa com o cone ainda
            # longe e piora quando ele chega perto (em curva a reta ajustada
            # desvia), entao: mediana das medidas, ignorando as de muito
            # perto assim que ja ha medidas boas.
            if (cone.altura <= config.CONE_MEDIDA_ALTURA_MAX or not self._w_obs):
                self._w_obs.append(w)
                if len(self._w_obs) > 25:
                    self._w_obs.pop(0)
                self._w_mem = float(np.median(self._w_obs))
            self.conf = 1.0
            return self._w_mem, True
        if self._w_mem is not None:
            # a continua saiu de vista (curva, oclusao): usa a ultima medida
            self.conf = 1.0
            return self._w_mem, True
        if lane is None or not lane.hw_px:
            return None, False
        w = g
        if w is None:
            return None, False
        # confianca sobe com o numero de estimativas (o cone e fixo: a
        # mediana de varias e bem melhor que uma so)
        n = len(self._w_geo)
        self.conf = max(config.CONE_CONF_GEO, min(1.0, 0.4 + 0.2 * n))
        return w, False

    def _w_geometrico(self, cone, lane, grava):
        """Posicao lateral do cone (cm do centro da pista, + = direita) pela
        geometria: onde o carro esta + o quanto o cone esta ao lado dele.
        Mediana das ultimas estimativas (o cone e fixo)."""
        if lane is None or not lane.hw_px or lane.lane_center_x is None:
            return None
        lw, cw = config.PISTA_LARGURA_CM, config.CONE_LARGURA_CM
        car_x = config.CAMERA_RES[0] * config.CAR_CENTER_PCT
        w_car = -((lane.lane_center_x - car_x) / lane.hw_px) * (lw / 2.0)
        w = w_car + cone.lat * cw
        if cone.altura < config.CONE_MEDIDA_ALTURA_MIN:
            return w                       # longe demais: so referencia
        if grava:
            self._w_geo.append(w)
            if len(self._w_geo) > 25:
                self._w_geo.pop(0)
            return float(np.median(self._w_geo))
        return float(np.median(self._w_geo + [w]))

    def _offset_posicao(self, cone, lane):
        """Deslocamento (unidades do erro da pista) que poe o ALVO do carro a
        uma folga fixa do cone, em cm. None se nao ha geometria confiavel.

        Em vez de mirar o cone (que gira o carro cada vez mais forte quanto
        mais perto), diz ONDE na faixa o carro deve estar. O seguidor de faixa
        ja sintonizado leva o carro ate la e, sem o deslocamento, traz de
        volta. E um PISO (so empurra pra longe do cone, nunca pra perto)."""
        s = getattr(lane, "sinal_continuo", 0.0) if lane is not None else 0.0
        if lane is None or not lane.hw_px or abs(s) < 0.5:
            return None
        self._s_cont = s
        w_cone, _ = self._w_cone(cone, lane)
        if w_cone is None:
            return None

        lw = config.PISTA_LARGURA_CM
        folga = config.CONE_FOLGA_CM
        w_t = (w_cone - folga) if self.lado > 0 else (w_cone + folga)

        # limites fisicos: nao passa da CONTINUA, e nao vai longe demais
        # alem da pontilhada (a faixa do outro lado tem uma largura so)
        margem_cont = config.CONE_MARGEM_CONT_CM
        alem_pont = 0.8 * config.CONE_CRUZAR_FRAC * lw
        if s > 0:       # continua a direita
            w_min, w_max = -(lw / 2.0 + alem_pont), lw / 2.0 - margem_cont
        else:
            w_min, w_max = -(lw / 2.0 - margem_cont), lw / 2.0 + alem_pont
        w_t = max(w_min, min(w_max, w_t))
        self.w_cone_cm, self.w_alvo_cm = w_cone, w_t
        self._faixa_alvo_outra = (w_t < -lw / 2.0 - 2.0) if s > 0 else (w_t > lw / 2.0 + 2.0)

        # onde o alvo da pista (ja com o vies da pontilhada) esta hoje
        w_now = (lane.center_x - lane.lane_center_x) / lane.hw_px * (lw / 2.0)
        if self.lado > 0:
            off_cm = min(0.0, w_t - w_now)        # so empurra pra ESQUERDA
        else:
            off_cm = max(0.0, w_t - w_now)        # so empurra pra DIREITA
        metade = config.CAMERA_RES[0] / 2.0
        return off_cm / (lw / 2.0) * lane.hw_px / metade

    def _off_base(self, lane):
        """Deslocamento (unidades do erro) que mantem o carro no centro da
        OUTRA faixa depois que desviou pra la: o carro NAO volta sozinho pra
        faixa de antes, so quando um cone na faixa dele pedir. 0 se esta na
        faixa de sempre."""
        if (not config.CONE_FAIXA_PERSISTE or not self.faixa_outra or lane is None
                or not lane.hw_px):
            return 0.0
        s = getattr(lane, "sinal_continuo", 0.0)
        if abs(s) < 0.5:
            return 0.0
        lw = config.PISTA_LARGURA_CM
        w_outra = -0.95 * lw if s > 0 else 0.95 * lw      # centro da outra faixa
        w_now = (lane.center_x - lane.lane_center_x) / lane.hw_px * (lw / 2.0)
        off_cm = (min(0.0, w_outra - w_now) if s > 0 else max(0.0, w_outra - w_now))
        metade = config.CAMERA_RES[0] / 2.0
        return off_cm / (lw / 2.0) * lane.hw_px / metade

    def _deslocamento(self, cone, e_pista):
        """Quanto ACIMA do erro da pista o alvo precisa ir pra cumprir a folga.
        A folga e medida em larguras de cone: pede pouco com o cone longe e
        muito com ele colado. Descontar e_pista e o que faz a pista nunca
        ganhar do cone (ver docstring do modulo)."""
        folga = (config.CONE_FOLGA_CM / config.CONE_LARGURA_CM) * cone.larg
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

    def _sem_cone(self, agora, lane=None):
        """Cone sumiu (saiu do campo de visao ou falhou a deteccao)."""
        base = self._off_base(lane)
        self.off_base = base
        if self._t_visto is None:
            self.ativo = False
            return base

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
            self._lado_medido = False
            self._w_mem = None
            self._w_obs = []
            self._w_geo = []
            self._w_lane = None
            self._w_lane_obs = []
            self.prox = 0.0
            self.larg = 0.0
            self.ativo = False
            self.na_outra = False
            self._t_outra = None
            return base
        return base + (self._alvo - base) * f
