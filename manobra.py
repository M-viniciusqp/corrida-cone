"""
manobra.py - PARA, ANALISA e so entao DESVIA.

Em vez de desviar andando (e errar o lado, ou perder a pista no meio), o
carro faz assim:

  SEGUIR   anda normal seguindo a faixa. O desvio fica so OBSERVANDO o cone.
  FREAR    o cone chegou na distancia de gatilho (CONE_GATILHO_ALTURA): zera os
           motores e ESPERA o carro parar de verdade (a inercia continua
           empurrando um pouco). "Parou" = o cone nao muda de tamanho/posicao
           na imagem e a camera da pista nao ve o chao se mexendo, por alguns
           quadros seguidos (com tempo minimo e maximo).
  ANALISAR carro parado: mede de novo a posicao do cone, so com leituras do
           carro PARADO (sem giro nem balanco), e escolhe o lado.
  DESVIAR  libera o desvio: anda e desvia pro lado escolhido. Termina quando o
           desvio acaba (o cone ja passou e o carro voltou pra faixa).

Nada aqui conhece motor/camera: recebe medidas e devolve se o carro deve ficar
parado. Main.py e o simulador usam a mesma classe.
"""

import config

SEGUIR, FREAR, ANALISAR, DESVIAR = "SEGUIR", "FREAR", "ANALISAR", "DESVIAR"


class Manobra:
    def __init__(self):
        self.ativa = bool(getattr(config, "CONE_PARA_E_ANALISA", False))
        self.estado = SEGUIR
        self.t_estado = 0.0
        self.t_ref = 0.0            # quando comecou a parar (pro log)
        self.calmo = 0              # quadros seguidos "sem se mexer"
        self.vistos = 0             # quadros novos seguidos com o cone no gatilho
        self.t_ultimo_gatilho = -1e9
        self._cone_ant = None       # (altura, cx) do quadro novo anterior
        self._rearmado = False      # em DESVIAR: o cone antigo ja saiu de vista
        self._t_sem_cone = None
        self._alt_max = 0.0         # maior altura do cone visto neste desvio
        self._fora_da_pista = False  # o cone visto esta fora da pista: ignora
        self.seq = False            # esta parada e de um cone EM SEQUENCIA (tempos mais curtos)
        self.ignorando = False      # o cone visto e de OUTRA faixa: nao para por ele
        self._ign_alt = 0.0
        self.ultimo = ""            # texto da ultima decisao (pro log)
        self.mov_cone = None        # ultimas medidas (pro log)
        self.mov_pista = None

    @property
    def parado(self):
        """True = os motores devem ficar em zero."""
        return self.estado in (FREAR, ANALISAR)

    # ------------------------------------------------------------------
    def _gatilho(self, cone):
        if self._fora_da_pista:
            return False
        return (cone is not None and cone.found and
                abs(cone.cx) < config.CONE_PARAR_X and
                (cone.altura >= config.CONE_GATILHO_ALTURA or cone.toca_base))

    def _muda(self, estado, agora):
        self.estado = estado
        self.t_estado = agora
        self.calmo = 0

    def update(self, agora, cone, cone_novo, mov_pista, desvio, lane=None):
        """Chamar a cada quadro. cone_novo = chegou quadro novo da camera de
        cone. mov_pista = quanto a imagem da pista mudou desde o quadro
        anterior (None se nao ha). Devolve o estado."""
        if not self.ativa:
            desvio.liberado = True
            return self.estado
        self.mov_pista = mov_pista
        # cone fora da pista (do outro lado das linhas externas): nao e obstaculo
        self._fora_da_pista = (cone is not None and cone.found and lane is not None
                               and not desvio.no_caminho(cone, lane))

        # quanto o cone mexeu na imagem desde o quadro novo anterior
        if cone_novo:
            if cone is not None and cone.found:
                atual = (cone.altura, cone.cx)
                if self._cone_ant is not None:
                    self.mov_cone = (abs(atual[0] - self._cone_ant[0]),
                                     abs(atual[1] - self._cone_ant[1]))
                else:
                    self.mov_cone = None
                self._cone_ant = atual
            else:
                self._cone_ant = None
                self.mov_cone = None
            if self.estado == SEGUIR:
                self.vistos = self.vistos + 1 if self._gatilho(cone) else 0

        if self.estado == SEGUIR:
            desvio.liberado = False
            if self.ignorando:
                # cone da OUTRA faixa: nao para, mas o desvio segue livre pra so
                # se afastar um pouco dele (centraliza o carro na faixa) e, se a
                # classificacao errou e ele estiver no caminho, ainda desvia
                desvio.liberado = True
                if cone is None or not cone.found:
                    if self._t_sem_cone is None:
                        self._t_sem_cone = agora
                    elif agora - self._t_sem_cone >= config.CONE_REARME_S:
                        self.ignorando = False
                else:
                    self._t_sem_cone = None
                    self._ign_alt = max(self._ign_alt, cone.altura)
                    if self._ign_alt > 0.3 and cone.altura < 0.5 * self._ign_alt:
                        self.ignorando = False      # apareceu OUTRO cone, mais longe
                self.vistos = 0
                return self.estado
            if (self.vistos >= config.CONE_GATILHO_QUADROS and
                    agora - self.t_ultimo_gatilho > config.CONE_COOLDOWN_S):
                desvio.reset(manter_faixa=True)   # cone novo: esquece o desvio anterior, mas fica na faixa
                self.t_ref = agora
                self.seq = (agora - self.t_ultimo_gatilho) < config.CONE_SEQ_JANELA_S
                self._muda(FREAR, agora)
                self.vistos = 0
            return self.estado

        if self.estado == FREAR:
            desvio.liberado = False
            dt = agora - self.t_estado
            if self._parado_agora():
                self.calmo += 1
            else:
                self.calmo = 0
            min_s = config.CONE_SEQ_PARADA_MIN_S if self.seq else config.CONE_PARADA_MIN_S
            if ((dt >= min_s and
                    self.calmo >= config.CONE_PARADA_CALMO_N) or
                    dt >= config.CONE_PARADA_MAX_S):
                # carro parado: comeca a medir do zero (as leituras da
                # aproximacao eram de longe e com o carro balancando)
                desvio.reiniciar_medidas()
                self._muda(ANALISAR, agora)
            return self.estado

        if self.estado == ANALISAR:
            desvio.liberado = False
            analise_s = config.CONE_SEQ_ANALISE_S if self.seq else config.CONE_ANALISE_S
            if agora - self.t_estado >= analise_s:
                if cone is not None and cone.found:
                    rel, txt = desvio.classificar_parado(cone, lane)
                    self.ultimo = txt
                    if not rel:
                        # NAO esta na minha faixa: segue em frente, sem desviar
                        self.ignorando = True
                        self._ign_alt = cone.altura
                        self._t_sem_cone = None
                        self.t_ultimo_gatilho = agora
                        desvio.reset(manter_faixa=True)
                        self._muda(SEGUIR, agora)
                        return self.estado
                    desvio.liberado = True
                    desvio.congelar_decisao(cone, lane)
                    self._rearmado = False
                    self._t_sem_cone = None
                    self._alt_max = cone.altura
                    self._muda(DESVIAR, agora)
                else:
                    # perdeu o cone enquanto estava parado: nao tem o que
                    # desviar, volta a seguir a faixa
                    self.t_ultimo_gatilho = agora
                    self._muda(SEGUIR, agora)
            return self.estado

        # DESVIAR
        desvio.liberado = True
        visto = cone is not None and cone.found
        if not visto:
            if self._t_sem_cone is None:
                self._t_sem_cone = agora
            elif agora - self._t_sem_cone >= config.CONE_REARME_S:
                self._rearmado = True       # o cone antigo ja ficou pra tras
        else:
            self._t_sem_cone = None
            # o cone que estamos passando so CRESCE na imagem; um bem menor
            # que o maior ja visto e OUTRO cone (mais longe): pode parar por ele
            self._alt_max = max(self._alt_max, cone.altura)
            if self._alt_max > 0.3 and cone.altura < 0.5 * self._alt_max:
                self._rearmado = True
        if self._rearmado and cone_novo:
            self.vistos = self.vistos + 1 if self._gatilho(cone) else 0
            if self.vistos >= config.CONE_GATILHO_QUADROS:
                # outro cone no gatilho: para de novo
                desvio.reset(manter_faixa=True)
                self.t_ref = agora
                self.seq = True
                self._muda(FREAR, agora)
                self.vistos = 0
                return self.estado
        fim = (not desvio.ativo and abs(desvio.offset - desvio.off_base) < 0.05)
        if fim or agora - self.t_estado > config.CONE_DESVIO_MAX_S:
            self.t_ultimo_gatilho = agora
            desvio.liberado = False
            desvio.descongelar()
            self._muda(SEGUIR, agora)
        return self.estado

    def _parado_agora(self):
        """O carro esta parado? Cone estavel na imagem E chao da pista quieto."""
        if self.mov_cone is not None:
            d_alt, d_x = self.mov_cone
            if d_alt > config.CONE_PARADO_D_ALT or d_x > config.CONE_PARADO_D_X:
                return False
        if self.mov_pista is not None and self.mov_pista > config.CONE_PARADO_DIFF:
            return False
        return True
