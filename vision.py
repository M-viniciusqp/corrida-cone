"""
VISAO SEM CALIBRACAO (scanline / varredura por linhas)
------------------------------------------------------
Nao usa warp, nao usa calibrate_warp.py, nao precisa medir nada.

Ideia: em vez de "achar a faixa", varre N linhas horizontais da imagem.
Em cada linha procura a borda branca mais proxima a ESQUERDA do carro e a
mais proxima a DIREITA. O meio dessas duas = centro da pista naquela altura.

O pulo do gato (o que substitui a calibracao):
a meia-largura da pista em pixels varia com a altura da imagem por causa da
perspectiva. Em vez de voce medir isso na mao, o codigo APRENDE sozinho:
nas linhas onde achou os dois lados, ele guarda (y, meia_largura) e ajusta
uma reta. Nas linhas onde so achou um lado (tipico do pontilhado), ele usa
essa reta pra estimar onde estaria o outro lado.

API identica a vision.LaneVision -> main.py e controller.py nao mudam.
"""

import cv2
import numpy as np
import config


class LaneResult:
    __slots__ = ("error", "heading", "curvature", "found_left", "found_right",
                 "valid", "center_x", "debug_view", "debug_mask",
                 "motivo", "n_pontos", "lado_continuo",
                 "lane_center_x", "hw_px", "sinal_continuo",
                 "cone_w_cm", "cone_px", "error_u")

    def __init__(self):
        self.error = 0.0
        self.heading = 0.0
        self.curvature = 0.0
        self.found_left = False
        self.found_right = False
        self.valid = False
        self.center_x = 0.0
        self.debug_view = None
        self.debug_mask = None
        self.motivo = "ok"     # por que perdeu, quando valid=False
        self.n_pontos = 0      # quantas linhas varridas deram ponto valido
        self.lado_continuo = None   # 'esquerda' | 'direita' | None (ainda indefinido)
        # usados pelo desvio de cones (cones.py): onde esta o centro da pista
        # (sem o vies de faixa) e quanto vale a meia-largura ali, em pixels.
        # Dao os limites laterais pra o desvio nunca empurrar o carro pra fora.
        self.lane_center_x = 0.0
        self.hw_px = None
        # -1..+1: +1 = faixa continua a DIREITA (pontilhada a esquerda).
        # O desvio de cones usa isso pra saber qual borda da pista pode ser
        # cruzada (a pontilhada) e qual nao (a continua).
        self.sinal_continuo = 0.0
        # cone visto PELA CAMERA DA PISTA: posicao lateral em cm a partir do
        # centro da faixa direita (+ = direita), comparada ao centro da faixa
        # NA MESMA FILEIRA da imagem (imune ao giro do carro e a curva). None
        # = nao viu cone / nao deu pra medir.
        self.cone_w_cm = None
        self.cone_px = None        # (x, y_base) so pra HUD
        # erro da pista SEM o corte em +-1 (suavizado igual ao error). O corte
        # esconde o quanto o carro passou do alvo: com o carro bem fora do
        # centro o erro "satura" em 1.0 e, somado ao desvio de cone (negativo),
        # o resultado e fraco demais pra trazer o carro de volta.
        self.error_u = 0.0


class LaneVision:
    def __init__(self):
        self.W, self.H = config.CAMERA_RES
        self._k_open = np.ones((3, 3), np.uint8)
        self._k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        # kernel do top-hat: precisa ser MAIOR que a largura da faixa,
        # senao ele apaga a propria faixa junto com o fundo.
        k = config.LINE_MAX_WIDTH_PX * 2 + 1
        self._k_line = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

        # modelo aprendido: meia_largura(y) = a*y + b   (em pixels)
        self.hw_model = None
        self.smooth_error = 0.0
        self.smooth_error_u = 0.0
        self.last_center = self.W * config.CAR_CENTER_PCT
        self.lost_frames = 0
        self.last_thr = 0.0
        self.white_frac = 0.0
        self.ref_e = None
        self.ref_d = None
        self.sat_frames = 0
        self.smooth_heading = 0.0

        # ---- deteccao automatica de qual lado e a faixa CONTINUA ----
        # Ideia: a faixa pontilhada tem BURACOS. Rodando varias linhas de
        # varredura (scanlines) por frame, o lado pontilhado vai ficar sem
        # deteccao em uma fracao das linhas (caiu no vao entre tracos),
        # enquanto o lado continuo e detectado em quase todas. Isso nao
        # depende de qual lado (esquerda/direita da imagem) e qual faixa -
        # funciona igual nos dois sentidos da pista.
        self.smooth_taxa_e = 0.5    # taxa de deteccao suavizada, lado esquerdo
        self.smooth_taxa_d = 0.5    # taxa de deteccao suavizada, lado direito
        self.lado_continuo = None   # 'esquerda' | 'direita' | None -- so texto/HUD
        self._sinal_continuo = 0.0  # -1..+1, o que REALMENTE define o vies (continuo)

        # TRAVA da decisao. Sem isso, um trecho ruim no meio da execucao
        # (rastreio perdendo um lado por causa de esterco violento, camera
        # engasgando, etc) derruba a taxa de deteccao DAQUELE lado por um
        # motivo que nao tem nada a ver com o padrao real da faixa -- e o
        # codigo interpretava isso como troca de lado, o alvo pulava pro
        # lado oposto da pista, e o carro saia. Uma vez confiante, a decisao
        # trava pro resto da execucao e para de reagir a chacoalhao.
        self._frames_consistentes = 0
        self._ultimo_sinal_consistente = 0   # -1, 0 ou +1
        self._sinal_travado = None    # None ate travar; depois fica -1.0 ou +1.0

        # Main liga isso durante o desvio de cone: o carro cruza a pontilhada
        # e, por um tempo, as taxas de deteccao nao dizem nada sobre qual
        # lado e a continua. Enquanto True, o aprendizado do lado fica parado.
        self.congelar_lado = False

        # carro em cima da faixa OPOSTA (depois de cruzar a pontilhada)?
        self._prev_media = None     # centro da faixa (px) no quadro anterior
        self._opp_janela = 0        # quadros restantes em que a correcao vale
        self.faixa_oposta = False   # exposto p/ log/HUD

        # lado da continua conhecido de antemao (config): ja nasce travado
        fixo = getattr(config, "LADO_CONTINUO_FIXO", None)
        if fixo in ("direita", "esquerda"):
            self._sinal_travado = 1.0 if fixo == "direita" else -1.0
            self._sinal_continuo = self._sinal_travado
            self.lado_continuo = fixo

    # ------------------------------------------------------------------
    def threshold(self, roi):
        """
        Segmentacao INVARIANTE A ILUMINACAO.

        Nao pergunta "esse pixel e claro?" (isso depende da luz do ambiente e
        e o que fazia o Otsu eleger o chao inteiro no escuro). Pergunta
        "esse pixel e mais claro que a VIZINHANCA dele?" - o que continua
        verdade tanto no sol quanto na sombra.

        Operador: TOP-HAT = imagem - abertura(imagem).
        A abertura com um kernel maior que a faixa apaga a faixa e deixa so o
        fundo/gradiente de luz. Subtraindo, sobra so estrutura clara e fina.
        """
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]

        gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, self._k_line)

        # limiar automatico: percentil do proprio frame, nao numero fixo
        thr = max(float(config.TOPHAT_MIN),
                  float(np.percentile(tophat, config.TOPHAT_PERCENTIL)) * config.TOPHAT_FATOR)

        # trava de seguranca: faixa nunca ocupa 1/4 da tela.
        # Se ocupou, o limiar estava baixo demais -> sobe e tenta de novo.
        mask = None
        for _ in range(6):
            mask = ((tophat >= thr) &
                    (S <= config.SAT_MAX) &
                    (V >= config.V_MIN_ABS)).astype(np.uint8) * 255
            if mask.mean() / 255.0 <= config.MAX_WHITE_FRAC:
                break
            thr *= 1.3

        self.last_thr = thr
        self.white_frac = mask.mean() / 255.0

        # tira cones da mascara: o plastico brilhante tem reflexos brancos e
        # pouco saturados que passavam por "faixa" e bagunÃ§avam o erro bem na
        # hora do desvio. Dilata um pouco pra pegar o reflexo na borda.
        if config.CONE_PISTA_DILATA_PX > 0:
            s_min, v_min = config.CONE_S_MIN, config.CONE_V_MIN
            red = cv2.bitwise_or(
                cv2.inRange(hsv, (0, s_min, v_min),
                            (config.CONE_H_BAIXO, 255, 255)),
                cv2.inRange(hsv, (config.CONE_H_ALTO, s_min, v_min),
                            (179, 255, 255)))
            if red.any():
                k = config.CONE_PISTA_DILATA_PX
                red = cv2.dilate(red, np.ones((k, k), np.uint8))
                mask[red > 0] = 0

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._k_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._k_close)
        return mask

    @staticmethod
    def _runs(row):
        """Retorna [(x_ini, x_fim)] dos trechos brancos da linha."""
        idx = np.flatnonzero(row)
        if idx.size == 0:
            return []
        cortes = np.flatnonzero(np.diff(idx) > 1)
        grupos = np.split(idx, cortes + 1)
        return [(int(g[0]), int(g[-1])) for g in grupos
                if len(g) >= config.SCAN_MIN_RUN_PX]

    # ------------------------------------------------------------------
    def process(self, frame, debug=False):
        res = LaneResult()

        y0 = int(self.H * config.SCAN_TOP_PCT)      # corta ceu/fundo
        roi = frame[y0:, :]
        mask = self.threshold(roi)
        roi_h = mask.shape[0]

        ys = np.linspace(roi_h - 5, 5, config.SCAN_ROWS).astype(int)

        centros = []          # (y_roi, x_centro, confianca)
        pares = []            # (y_roi, meia_largura) -> alimenta o modelo
        car_x = self.W * config.CAR_CENTER_PCT

        # Referencias de rastreio: onde cada faixa estava no frame anterior.
        ref_e, ref_d = self.ref_e, self.ref_d
        jan = config.SCAN_TRACK_WINDOW

        # Referencias pro PROXIMO quadro: a posicao das linhas na fileira de
        # BAIXO (a primeira valida), nao na de cima. O proximo quadro comeca a
        # busca pela fileira de baixo; com as linhas inclinadas (carro virado
        # ou fora do centro) a posicao de cima fica a 100+ px da de baixo,
        # fora da janela de rastreio, e a linha era perdida ou trocada pela
        # vizinha (a pontilhada capturada como se fosse a continua).
        ref_bottom = None

        # contadores pra deteccao automatica do lado continuo (ver __init__)
        n_rows_validas = 0
        n_hit_e = 0
        n_hit_d = 0

        # ys vem de BAIXO pra CIMA. Perto do carro a faixa e confiavel, entao a
        # identidade (esquerda/direita) e definida ali e PROPAGADA pra cima
        # seguindo a curva. E isso que impede a faixa de "trocar de lado"
        # quando ela cruza o centro do quadro numa curva forte.
        for y in ys:
            runs = self._runs(mask[y])
            if not runs:
                continue

            n_rows_validas += 1
            xe = xd = None
            i_e = i_d = -1

            if ref_e is None or ref_d is None:
                s_c = self._sinal_continuo
                todas = list(enumerate(runs))
                if abs(s_c) > 0.5 and len(todas) >= 1:
                    # sem historico, mas se SABE qual lado e a continua: ancora
                    # nela (a mais externa daquele lado) e pega a vizinha. O
                    # corte pelo centro do carro erra quando o carro esta do
                    # outro lado da pontilhada (desvio de cone): pegaria a
                    # linha externa do outro lado como "esquerda".
                    if s_c > 0:
                        i_d, r = max(todas, key=lambda p: p[1][1]); xd = r[0]
                        viz = [(i, q) for i, q in todas if q[1] < xd - config.SCAN_MIN_RUN_PX]
                        if viz:
                            i_e, q = max(viz, key=lambda p: p[1][1]); xe = q[1]
                    else:
                        i_e, r = min(todas, key=lambda p: p[1][0]); xe = r[1]
                        viz = [(i, q) for i, q in todas if q[0] > xe + config.SCAN_MIN_RUN_PX]
                        if viz:
                            i_d, q = min(viz, key=lambda p: p[1][0]); xd = q[0]
                else:
                    # sem historico: unica vez que usa o centro do carro como corte
                    esq = [(i, r) for i, r in todas if r[1] < car_x]
                    dirr = [(i, r) for i, r in todas if r[0] > car_x]
                    if esq:
                        i_e, r = max(esq, key=lambda p: p[1][1]); xe = r[1]
                    if dirr:
                        i_d, r = min(dirr, key=lambda p: p[1][0]); xd = r[0]
            else:
                # rastreio local: trecho branco mais proximo de onde essa
                # faixa estava na linha de baixo
                ce = [(i, r) for i, r in enumerate(runs) if abs(r[1] - ref_e) <= jan]
                cd = [(i, r) for i, r in enumerate(runs) if abs(r[0] - ref_d) <= jan]
                if ce:
                    i_e, r = min(ce, key=lambda p: abs(p[1][1] - ref_e)); xe = r[1]
                if cd:
                    i_d, r = min(cd, key=lambda p: abs(p[1][0] - ref_d)); xd = r[0]

                # as duas referencias agarraram o MESMO trecho: e uma faixa so.
                # Fica com a mais proxima e descarta a outra.
                if i_e >= 0 and i_e == i_d:
                    if abs(xe - ref_e) <= abs(xd - ref_d):
                        xd, i_d = None, -1
                    else:
                        xe, i_e = None, -1

            if xe is not None:
                n_hit_e += 1
            if xd is not None:
                n_hit_d += 1

            # ---- achou os dois: linha de ouro, alimenta o modelo de largura
            if xe is not None and xd is not None:
                largura = (xd - xe) / 2.0
                ok = config.SCAN_HW_MIN < largura < config.SCAN_HW_MAX
                esperado = self._half_width(y) if ok else None
                # a largura do par tem que bater com a da faixa APRENDIDA.
                # Antes so conferia uma faixa absoluta (25..160 px), entao
                # "linha externa do outro lado + continua" (o dobro da
                # largura) passava, o centro virava a propria pontilhada e o
                # carro era puxado pra outra faixa depois de um desvio.
                if ok and esperado is not None and \
                        abs(largura - esperado) > config.SCAN_HW_TOL * esperado:
                    ok = False
                if ok:
                    centros.append((y, (xe + xd) / 2.0, 1.0))
                    pares.append((y, largura))
                    res.found_left = res.found_right = True
                    ref_e, ref_d = xe, xd
                    if ref_bottom is None:
                        ref_bottom = (ref_e, ref_d)
                    continue
                # par inconsistente: confia na faixa CONTINUA (se se sabe
                # qual e) e completa o outro lado pela largura aprendida
                if esperado is not None and abs(self._sinal_continuo) > 0.5:
                    if self._sinal_continuo > 0:
                        xe, i_e = None, -1
                    else:
                        xd, i_d = None, -1
                else:
                    continue

            if xe is None and xd is None:
                continue

            # ---- so um lado: completa com a meia-largura aprendida
            hw = self._half_width(y)
            if hw is None:
                # ainda sem modelo: nao inventa centro, mas mantem o rastreio vivo
                if xe is not None:
                    ref_e = xe
                if xd is not None:
                    ref_d = xd
                continue

            if xe is not None:
                centros.append((y, xe + hw, 0.6))
                res.found_left = True
                ref_e, ref_d = xe, xe + 2 * hw
            else:
                centros.append((y, xd - hw, 0.6))
                res.found_right = True
                ref_d, ref_e = xd, xd - 2 * hw
            if ref_bottom is None:
                ref_bottom = (ref_e, ref_d)

        # guarda a base do rastreio pro proximo frame
        if ref_bottom is not None:
            self.ref_e, self.ref_d = ref_bottom
        else:
            self.ref_e, self.ref_d = ref_e, ref_d

        # ---- CONTINUIDADE do centro da faixa (so durante/apos um desvio)
        # Com o carro do outro lado da pontilhada, a camera ve [linha externa
        # do outro lado] + [pontilhada], uma dupla com a MESMA largura de uma
        # faixa: o centro bruto cai no centro da OUTRA faixa, 2*hw pra um
        # lado. Um centro de faixa nao pode pular uma faixa inteira em 33 ms,
        # entao entre {bruto, bruto +- 2*hw} fica o mais proximo do centro do
        # quadro anterior. Sem estatistica de qual linha e continua.
        if self.congelar_lado:
            self._opp_janela = config.FAIXA_OPOSTA_JANELA
        elif self._opp_janela > 0:
            self._opp_janela -= 1

        k_shift = 0
        if centros:
            cxs = np.array([c[1] for c in centros], dtype=float)
            cws = np.array([c[2] for c in centros], dtype=float)
            media0 = float(np.average(cxs, weights=cws))
            if (self._opp_janela > 0 and self._prev_media is not None
                    and self.hw_model is not None):
                hw_ref = float(np.polyval(self.hw_model, roi_h * 0.5))
                hw_ref = max(config.SCAN_HW_MIN, min(config.SCAN_HW_MAX, hw_ref))
                sgn = 1.0 if self._sinal_continuo >= 0 else -1.0
                melhor_d = abs(media0 - self._prev_media)
                for k in (-1, 1):
                    d = abs(media0 + k * 2.0 * hw_ref - self._prev_media)
                    # exige ganho CLARO (histerese): so troca se a hipotese
                    # deslocada for bem mais coerente com o quadro anterior
                    if d < melhor_d - 0.5 * hw_ref and d < config.FAIXA_OPOSTA_GATE * hw_ref:
                        melhor_d, k_shift = d, k
            if k_shift != 0:
                novos = []
                for (yy, xx, cc) in centros:
                    hw_y = self._half_width(yy)
                    if hw_y is None:
                        hw_y = float(np.polyval(self.hw_model, yy))
                    novos.append((yy, xx + k_shift * 2.0 * hw_y, cc))
                centros = novos
                media0 += k_shift * 2.0 * hw_ref
            self._prev_media = media0
        self.faixa_oposta = k_shift != 0

        # ---- deteccao automatica do lado continuo x pontilhado ----
        # ANTES: decisao binaria (vira "esquerda" OU "direita" de uma vez).
        # Isso fazia o alvo SALTAR de +mag*hw pra -mag*hw no frame exato em
        # que a decisao virava -- foi o que tirou o carro da pista. Agora
        # self._sinal_continuo e um numero continuo entre -1 e +1: perto de
        # 0 quando os dois lados estao parecidos (sem bias, fica centralizado
        # -- o mais seguro quando ha duvida), e so chega em +-1 quando um
        # lado esta CLARAMENTE mais solido que o outro. Como ele nasce de
        # smooth_taxa_e/d (que ja e um EMA), a transicao de um lado pro
        # outro e sempre gradual, nunca um degrau.
        if n_rows_validas >= config.SCAN_MIN_ROWS and not self.congelar_lado:
            taxa_e = n_hit_e / n_rows_validas
            taxa_d = n_hit_d / n_rows_validas
            b = config.LADO_CONTINUO_SUAVIZACAO
            self.smooth_taxa_e = b * self.smooth_taxa_e + (1 - b) * taxa_e
            self.smooth_taxa_d = b * self.smooth_taxa_d + (1 - b) * taxa_d

            diff = self.smooth_taxa_e - self.smooth_taxa_d

            if self._sinal_travado is not None:
                # JA TRAVADO: ignora qualquer flutuacao de taxa daqui pra
                # frente. Um trecho ruim no meio do percurso nao pode mais
                # virar a decisao.
                self._sinal_continuo = self._sinal_travado
            else:
                # positivo = precisa deslocar o alvo pra DIREITA (continua e a direita)
                self._sinal_continuo = max(-1.0, min(
                    1.0, -diff / config.LADO_CONTINUO_MARGEM))

                # conta frames CONSISTENTES (mesmo lado, com folga confortavel
                # acima da margem normal) pra decidir travar. Qualquer
                # inconsistencia zera a contagem -- so trava com sinal limpo.
                if abs(diff) > config.LADO_CONTINUO_TRAVA_MARGEM:
                    # mesmo sinal que _sinal_continuo usa (-diff/MARGEM)
                    sinal_atual = 1 if diff < 0 else -1
                    if sinal_atual == self._ultimo_sinal_consistente:
                        self._frames_consistentes += 1
                    else:
                        self._ultimo_sinal_consistente = sinal_atual
                        self._frames_consistentes = 1
                else:
                    self._frames_consistentes = 0
                    self._ultimo_sinal_consistente = 0

                if self._frames_consistentes >= config.LADO_CONTINUO_TRAVA_FRAMES:
                    self._sinal_travado = float(self._ultimo_sinal_consistente)
                    self._sinal_continuo = self._sinal_travado
                    # trava o ROTULO tambem -- ele deve bater com a decisao
                    # real, nao ficar recalculando por conta propria depois
                    self.lado_continuo = "direita" if self._sinal_travado > 0 else "esquerda"

            # rotulo so pra telemetria/HUD -- so atualiza enquanto AINDA NAO
            # travou. Depois de travado, o bloco acima ja fixou o rotulo e
            # este trecho e pulado (fica so o texto condizente com a trava).
            if self._sinal_travado is None:
                if diff > config.LADO_CONTINUO_MARGEM:
                    self.lado_continuo = "esquerda"
                elif -diff > config.LADO_CONTINUO_MARGEM:
                    self.lado_continuo = "direita"
                # diferenca pequena: mantem o ultimo rotulo, so texto mesmo
        res.lado_continuo = self.lado_continuo
        res.sinal_continuo = self._sinal_continuo

        # ---- atualiza o modelo de meia-largura
        if len(pares) >= 2:
            py = np.array([p[0] for p in pares], dtype=float)
            ph = np.array([p[1] for p in pares], dtype=float)
            novo = np.polyfit(py, ph, 1)
            # trava anti-deriva: o modelo so aceita ajuste pequeno por frame,
            # senao ele infla sozinho (cada frame usa a hw anterior pra formar
            # pares, que confirmam a hw inflada... e vira bola de neve)
            if self.hw_model is None:
                self.hw_model = novo
            else:
                self.hw_model = 0.9 * self.hw_model + 0.1 * novo
        elif len(pares) == 1 and self.hw_model is None:
            self.hw_model = np.array([0.0, pares[0][1]])

        # ---- sem informacao suficiente
        res.n_pontos = len(centros)
        if len(centros) < config.SCAN_MIN_CENTROS:
            self.lost_frames += 1
            res.valid = self.lost_frames <= config.MAX_FRAMES_PERDIDO
            res.error = self.smooth_error
            res.motivo = f"poucos_pontos({len(centros)}<{config.SCAN_MIN_CENTROS})"
            if debug:
                res.debug_view, res.debug_mask = self._draw(roi, mask, centros, None)
            return res

        self.lost_frames = 0

        cy = np.array([c[0] for c in centros], dtype=float)
        cx = np.array([c[1] for c in centros], dtype=float)
        cw = np.array([c[2] for c in centros], dtype=float)

        # peso extra pras linhas mais distantes (lookahead) mas sem ignorar o capo
        peso = cw * np.linspace(1.0, config.SCAN_PESO_LONGE, len(cy))[np.argsort(cy)[::-1].argsort()]

        media = float(np.average(cx, weights=peso))
        y_min, y_max = float(cy.min()), float(cy.max())
        span = y_max - y_min

        # NUNCA extrapolar: o alvo so pode ficar dentro da faixa de alturas que
        # o robo REALMENTE enxergou. Extrapolar parabola com poucos pontos
        # manda o alvo pra fora da tela (era o ponto verde colado na borda).
        y_look = roi_h * (1.0 - config.LOOKAHEAD_PCT)
        y_look = max(y_min, min(y_max, y_look))

        # grau 2 exige pontos suficientes E bem espalhados; senao, reta.
        grau = 2 if (len(cy) >= config.FIT_MIN_PTS_GRAU2 and
                     span >= roi_h * config.FIT_MIN_SPAN_PCT) else 1

        alvo = media
        raw_heading = 0.0
        try:
            fit = np.polyfit(cy, cx, grau, w=peso)
            cand = float(np.polyval(fit, y_look))
            # sanidade: o alvo tem que estar perto dos pontos observados
            if cx.min() - config.FIT_MARGEM <= cand <= cx.max() + config.FIT_MARGEM:
                alvo = cand
                if grau == 2:
                    raw_heading = 2 * fit[0] * y_look + fit[1]
                    res.curvature = float(abs(fit[0]))
                else:
                    raw_heading = fit[0]
        except (np.linalg.LinAlgError, ValueError):
            pass

        # TRAVA DE ESTABILIDADE DO HEADING.
        # O heading (dx/dy) alimenta o feed-forward do PID. Sem limite, um fit
        # ruim (poucos pontos, curva no limiar grau1/grau2) gera valores tipo
        # 3.4 quando o plausivel e proximo de +-1 - isso INVERTE o sinal do
        # esterco mesmo com o erro estavel. Grampeia e suaviza, igual ao erro.
        raw_heading = max(-config.HEADING_MAX, min(config.HEADING_MAX, raw_heading))
        self.smooth_heading = (config.HEADING_SUAVIZACAO * self.smooth_heading +
                               (1 - config.HEADING_SUAVIZACAO) * raw_heading)
        res.heading = self.smooth_heading

        # Se o alvo caiu FORA da imagem, o rastreio esta furado. Nao vale gerar
        # erro +-1 com confianca total: melhor declarar perdido.
        # (o centro da faixa pode cair um pouco fora da imagem quando o carro
        # esta deslocado ~meia faixa; o erro ja e limitado a +-1 mais abaixo)
        marg = config.ALVO_MARGEM_PCT * self.W
        if not (-marg <= alvo <= self.W + marg):
            self.ref_e = self.ref_d = None
            self.lost_frames += 1
            res.valid = self.lost_frames <= config.MAX_FRAMES_PERDIDO
            res.error = self.smooth_error
            res.center_x = float(np.clip(alvo, 0, self.W))
            res.motivo = f"alvo_fora_da_tela({alvo:.0f})"
            if debug:
                res.debug_view, res.debug_mask = self._draw(roi, mask, centros, None)
            return res

        # centro "puro" da pista (antes do vies) + meia-largura nessa altura
        res.lane_center_x = alvo
        res.hw_px = self._half_width(y_look)

        # ---- VIES DE FAIXA: andar mais perto da faixa CONTINUA que do centro.
        # Desloca o ALVO (nao o carro) em fracao da meia-largura da pista.
        # LANE_BIAS_FRAC agora e so a MAGNITUDE do vies (0..LANE_BIAS_LIMITE).
        # O SINAL (pra que lado desloca) vem de self.lado_continuo, que e
        # aprendido sozinho olhando qual lado tem buracos (pontilhado) e qual
        # nao tem (continuo) -- por isso funciona igual com a faixa continua
        # na esquerda ou na direita, sem precisar mudar nada no config.
        # Sempre escala com a largura real vista no frame -> funciona em
        # qualquer trecho, reto ou curva, sem virar valor fixo em pixel.
        alvo_u = alvo
        if config.LANE_BIAS_FRAC != 0.0:
            hw_look = self._half_width(y_look)
            if hw_look is not None:
                mag = min(config.LANE_BIAS_LIMITE, abs(config.LANE_BIAS_FRAC))
                # negativo aqui de proposito: self._sinal_continuo aponta pro
                # lado da CONTINUA, e o que voce quer e o oposto -- ficar mais
                # perto da PONTILHADA.
                alvo += -self._sinal_continuo * mag * hw_look
                alvo_u = alvo
                alvo = max(0.0, min(float(self.W), alvo))

        self.last_center = 0.6 * self.last_center + 0.4 * media

        car_x = self.W * config.CAR_CENTER_PCT
        raw = max(-1.0, min(1.0, (alvo - car_x) / (self.W / 2.0)))
        raw_u = max(-2.0, min(2.0, (alvo_u - car_x) / (self.W / 2.0)))

        a = config.ERRO_SUAVIZACAO
        self.smooth_error = a * self.smooth_error + (1 - a) * raw
        self.smooth_error_u = a * self.smooth_error_u + (1 - a) * raw_u
        res.error = self.smooth_error
        res.error_u = self.smooth_error_u

        # ---- TRAVA ANTI-SATURACAO
        # Erro colado em +-1 por muitos frames nao e curva: e rastreio perdido
        # (tipico de faixa trocando de lado). Melhor admitir que se perdeu do
        # que virar o esterco todo com confianca total.
        if abs(res.error) >= config.SAT_LIMITE:
            self.sat_frames += 1
        else:
            self.sat_frames = 0

        # (durante/logo apos um desvio de cone o carro fica de proposito longe
        # do centro da faixa e o erro satura: isso NAO e rastreio podre, e
        # zerar as referencias aqui fazia a visao se religar na linha errada)
        if (self.sat_frames > config.MAX_FRAMES_SATURADO and
                self._opp_janela == 0):
            self.ref_e = self.ref_d = None      # zera o rastreio e reengata
            self.sat_frames = 0
            self.smooth_error = 0.0             # nao carrega o erro podre adiante
            res.valid = False
            res.center_x = alvo
            res.motivo = "saturado"
            return res

        res.center_x = alvo
        res.valid = True

        if config.CONE_NA_PISTA:
            res.cone_w_cm, res.cone_px = self._cone_na_pista(frame, y0, centros)

        if debug:
            res.debug_view, res.debug_mask = self._draw(
                roi, mask, centros, (alvo, y_look, car_x))
        return res

    # ------------------------------------------------------------------
    def _cone_na_pista(self, frame, y0, centros):
        """Acha o cone (vermelho) na imagem da PISTA e diz onde ele esta na
        faixa: cm a partir do centro da faixa direita (+ = direita).

        Compara a base do cone com o centro da faixa e a meia-largura NA MESMA
        FILEIRA, entao nao depende do giro do carro nem da curva (a camera
        frontal mede o cone em relacao ao EIXO do carro, o que erra por ~30 cm
        com 10 graus de giro a 2 m)."""
        if self.hw_model is None or len(centros) < 3:
            return None, None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        s_min, v_min = config.CONE_S_MIN, config.CONE_V_MIN
        red = cv2.bitwise_or(
            cv2.inRange(hsv, (0, s_min, v_min), (config.CONE_H_BAIXO, 255, 255)),
            cv2.inRange(hsv, (config.CONE_H_ALTO, s_min, v_min), (179, 255, 255)))
        if not red.any():
            return None, None
        red = cv2.morphologyEx(red, cv2.MORPH_OPEN, self._k_open)
        n, _, st, _ = cv2.connectedComponentsWithStats(red, connectivity=8)
        melhor = None
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in st[i])
            if area < config.CONE_PISTA_AREA_MIN or w < 3 or h < 3:
                continue
            if h / float(w) > 5.0:          # fita/risco fino, nao e cone
                continue
            y_b = y + h - 1
            if melhor is None or y_b > melhor[1]:
                melhor = (x + w / 2.0, y_b, w)
        if melhor is None:
            return None, None
        x_c, y_b, w_px = melhor

        if y_b < config.CONE_PISTA_YB_MIN:
            return None, None                  # longe: a leitura e ruidosa demais
        # fileira da base do cone, nas coordenadas do ROI (as do modelo)
        yr = y_b - y0
        hw = float(np.polyval(self.hw_model, yr))
        if hw < config.SCAN_HW_MIN * 0.8 or hw > config.SCAN_HW_MAX:
            return None, None            # longe/perto demais pra medir bem
        # o cone tem a largura certa pra essa distancia? (rejeita vermelho
        # que nao e cone e cone cortado pela borda da imagem)
        esperado = 2.0 * hw * (config.CONE_LARGURA_CM / config.PISTA_LARGURA_CM)
        if not (0.35 * esperado <= w_px <= 1.8 * esperado):
            return None, None

        # 1) as LINHAS na mesma fileira da base do cone (a continua e solida:
        # sempre aparece). E o que nao depende de extrapolar a curva.
        w = self._w_por_linhas(frame, int(y_b), x_c, w_px, hw, red)

        # 2) senao, o centro da faixa por reta local, so se a base do cone
        # estiver perto da zona varrida (longe a curva faz a reta errar muito)
        if w is None:
            pts = sorted(centros, key=lambda c: c[0])[:6]
            ys = np.array([p[0] for p in pts], dtype=float)
            xs = np.array([p[1] for p in pts], dtype=float)
            ws = np.array([p[2] for p in pts], dtype=float)
            if yr < ys.min() - config.CONE_PISTA_EXTRAP_PX:
                return None, None
            if len(pts) >= 3 and (ys.max() - ys.min()) >= 12.0:
                try:
                    a_, b_ = np.polyfit(ys, xs, 1, w=ws)
                except (np.linalg.LinAlgError, ValueError):
                    return None, None
                x_l = a_ * yr + b_
            else:
                x_l = float(np.average(xs, weights=ws))
            w = (x_c - x_l) / hw * (config.PISTA_LARGURA_CM / 2.0)
        if abs(w) > 1.7 * config.PISTA_LARGURA_CM:
            return None, None            # fora da pista: nao e do caminho
        return float(w), (float(x_c), float(y_b))

    def _w_por_linhas(self, frame, y_b, x_c, w_px, hw, red):
        """Posicao do cone (cm do centro da faixa direita, + = direita) pelas
        linhas CONTINUAS vistas na fileira da base dele: a da direita (em
        +lw/2) e a externa da esquerda (em -1.5*lw). Nao precisa da
        pontilhada (tem buracos). None se nao achou, ou se as duas
        discordam."""
        lw = config.PISTA_LARGURA_CM
        H, W = frame.shape[:2]
        ya, yz = max(0, y_b - 1), min(H, y_b + 2)
        if yz - ya < 2:
            return None
        g = cv2.cvtColor(frame[ya:yz], cv2.COLOR_BGR2GRAY).astype(np.float32).mean(axis=0)
        kw = int(max(7, min(41, 0.5 * hw))) | 1
        gl = np.clip(g, 0, 255).astype(np.uint8).reshape(1, -1)
        th = cv2.morphologyEx(gl, cv2.MORPH_TOPHAT,
                              np.ones((1, kw), np.uint8))[0].astype(np.float32)
        # tira o vermelho (e a faixa branca do cone): so o chao conta
        ruim = (red[ya:yz].max(axis=0) > 0).astype(np.uint8).reshape(1, -1)
        ruim = cv2.dilate(ruim, np.ones((1, 9), np.uint8))[0] > 0
        th[ruim] = 0.0
        meio = 0.7 * w_px + 3.0
        xs = np.arange(W, dtype=np.float32)
        th[np.abs(xs - x_c) <= meio] = 0.0
        alc = 4.3 * hw
        th[np.abs(xs - x_c) > alc] = 0.0

        pico = float(th.max())
        if pico < config.TOPHAT_MIN * 0.8:
            return None
        thr = max(config.TOPHAT_MIN * 0.7, 0.4 * pico)
        on = th >= thr
        # trechos finos (linha), do tamanho de uma fita
        d = np.diff(np.concatenate(([0], on.astype(np.int8), [0])))
        ini, fim = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
        cand = [0.5 * (a + b) for a, b in zip(ini, fim) if 2 <= (b - a + 1) <= kw]
        if not cand:
            return None
        dir_ = [x for x in cand if x > x_c]
        esq_ = [x for x in cand if x < x_c]
        w_r = w_l = None
        # A linha mais EXTERNA de cada lado so e a continua se a continua
        # CABE na imagem desse lado (senao a "mais externa" pode ser a
        # pontilhada e o resultado erra uma faixa inteira, 29 cm).
        cabe_dir = (x_c + 3.1 * hw) < (W - 2)
        cabe_esq = (x_c - 3.1 * hw) > 2
        if dir_:
            x_r = max(dir_)                       # a mais externa
            if cabe_dir or (x_r - x_c) > 2.2 * hw:
                w_r = (lw / 2.0) * (1.0 - (x_r - x_c) / hw)
        if esq_:
            x_e = min(esq_)
            if cabe_esq or (x_c - x_e) > 2.2 * hw:
                w_l = -1.5 * lw + (x_c - x_e) / hw * (lw / 2.0)
        if w_r is not None and w_l is not None:
            if abs(w_r - w_l) > 0.4 * lw:
                return None
            return 0.5 * (w_r + w_l)
        return w_r if w_r is not None else w_l

    @property
    def em_desvio(self):
        """True durante e logo apos um desvio de cone (janela de ~3 s)."""
        return self._opp_janela > 0

    def _half_width(self, y):
        if self.hw_model is None:
            return None
        hw = float(np.polyval(self.hw_model, y))
        if hw < config.SCAN_HW_MIN or hw > config.SCAN_HW_MAX:
            return None
        return hw

    def _draw(self, roi, mask, centros, alvo):
        vis = roi.copy()
        for (y, x, conf) in centros:
            cor = (0, 255, 0) if conf >= 1.0 else (0, 165, 255)
            cv2.circle(vis, (int(x), int(y)), 3, cor, -1)
        if alvo is not None:
            ax, ay, car_x = alvo
            cv2.circle(vis, (int(car_x), vis.shape[0] - 8), 6, (0, 0, 255), -1)
            cv2.circle(vis, (int(ax), int(ay)), 7, (0, 255, 0), -1)
            cv2.line(vis, (int(car_x), vis.shape[0] - 8), (int(ax), int(ay)),
                     (255, 255, 255), 2)
        return vis, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
