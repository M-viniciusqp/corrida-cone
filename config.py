"""
config.py - todos os parametros do robo num lugar so.
Nenhum outro arquivo tem numero magico dentro.
"""

import numpy as np

# =====================================================================
# 1. HARDWARE (pinos BCM)
# =====================================================================
ENA_A = 5;  IN1_A = 6;  IN2_A = 13     # motor esquerdo
ENA_B = 17; IN3_B = 22; IN4_B = 27     # motor direito
SERVO_PIN = 18

# =====================================================================
# 2. SERVO (graus)
# SERVO_CENTRO nao e o meio geometrico: e o angulo em que a roda fica
# REALMENTE reta. Descubra com: python3 test_servo.py
# =====================================================================
SERVO_CENTRO   = 40
SERVO_MAX_ESQ  = 0
SERVO_MAX_DIR  = 90
SERVO_DEADBAND_DEG = 1.0   # so manda pulso se mudou mais que isso (mata o jitter)

STEER_INVERT = False

# =====================================================================
# 3. VELOCIDADE (0.0 a 1.0)
# =====================================================================
VELOCIDADE_BASE    = 0.25   # BAIXA de proposito pra validar o desvio de cones.
                            # Suba de 0.05 em 0.05 (ate 1.00) conforme ficar
                            # estavel. NUNCA deixe abaixo de VELOCIDADE_MINIMA
                            # (0.25), senao o speed_for() inverte. O freio por
                            # FPS baixo e o de curva fechada continuam ativos.
VELOCIDADE_MINIMA  = 0.22   # subiu de 0.35 -- piso normal de curva
VELOCIDADE_PERDIDO = 0.18   # perdeu a pista, andando devagar procurando

FPS_MINIMO_SEGURO = 15.0

# =====================================================================
# 4. CAMERA
# =====================================================================
CAMERA_RES = (480, 240)

# =====================================================================
# 5. SEGMENTACAO DA FAIXA BRANCA (top-hat, invariante a iluminacao)
# =====================================================================
LINE_MAX_WIDTH_PX = 18
TOPHAT_PERCENTIL  = 98
TOPHAT_FATOR      = 0.45
TOPHAT_MIN        = 18
SAT_MAX           = 90
V_MIN_ABS         = 70
MAX_WHITE_FRAC    = 0.22

# =====================================================================
# 6. VARREDURA (scanline)
# =====================================================================
SCAN_TOP_PCT    = 0.35
SCAN_ROWS       = 14
SCAN_MIN_RUN_PX = 4
SCAN_MIN_ROWS   = 6
SCAN_MIN_CENTROS = 3     # centros minimos pra aceitar a leitura. Com so a
                         # pontilhada visivel (carro em cima dela, a continua
                         # fora do campo) so ha centro onde tem traco: 3 a 6
                         # fileiras. Com 6 isso virava 'perdido' toda vez que
                         # o carro cruzava a pontilhada pra desviar.
SCAN_HW_MIN     = 25
SCAN_HW_MAX     = 160
SCAN_PESO_LONGE = 2.0
SCAN_HW_TOL     = 0.45   # par de linhas so vale se a largura bate com a da
                         # faixa aprendida (+-45%)
SCAN_TRACK_WINDOW = 95

ALVO_MARGEM_PCT = 0.5   # o alvo pode cair ate 50% da largura da imagem pra fora
                       # (antes: 0% -> 'alvo_fora_da_tela' zerava o rastreio quando
                       # o carro estava a ~meia faixa do centro, ex. cruzando a
                       # pontilhada pra desviar de cone)

HEADING_MAX          = 1.2
HEADING_SUAVIZACAO   = 0.55

FIT_MIN_PTS_GRAU2 = 6
FIT_MIN_SPAN_PCT  = 0.40
FIT_MARGEM        = 60

LOOKAHEAD_PCT = 0.55
CAR_CENTER_PCT = 0.50

# =====================================================================
# 6b. REACAO EM CURVA
# =====================================================================
FEEDFORWARD_HEADING   = 0.0
SPEED_TURN_PESO       = 1.2    # baixou um pouco de 1.3 -- menos freio pelo
                                # esterco atual, junto com o piso mais alto
SPEED_CURVATURA_PESO  = 75.0   # baixou de 90.0 -- mesma logica

CURVA_FECHADA_LIMIAR     = 0.004
CURVA_FECHADA_RANGE      = 0.014
VELOCIDADE_MINIMA_FECHADA = 0.20   # subiu de 0.32 -- piso mais alto so pra
                                    # curva bem fechada tambem

# =====================================================================
# 7. CONTROLE (PID)
# Ainda no valor conservador validado (KP=0.50). O procedimento de
# calibracao completo (secao 10) pode liberar um KP mais alto -- isso
# permite corrigir mais forte e rapido, o que deixa CURVAS mais rapidas
# sem precisar abrir mao de margem de seguranca. Vale fazer.
# =====================================================================
PID_KP = 0.62               # baixou de 1.25 -- o teste ESTATICO (carro na
                            # mao) nao revelou instabilidade, mas rodando de
                            # verdade na pista ele oscilou (erro -0.99 seguido
                            # de +0.50 no proximo instante) e saiu da faixa.
                            # 1.25 dinamico = Ku de verdade. Este valor e
                            # Ku*0.5, a regra classica de sempre. So volte a
                            # subir depois de confirmar ESTAVEL rodando na
                            # pista (nao so no teste parado).
PID_KI = 0.0
PID_KD = 0.12                # baixou de 0.15 -- estava dominando o comando
                            # em cima de erro pequeno, com KP=1.25
PID_D_SUAVIZACAO = 0.55      # subiu de 0.3 -- mais filtro na derivada, pra
                            # segurar o balanco sem voltar ao problema antigo
                            # de atraso empilhado (ERRO_SUAVIZACAO continua em
                            # 0.3, entao nao esta dobrando filtro pesado)
PID_I_LIMIT = 0.4
ERRO_SUAVIZACAO = 0.3
DEAD_ZONE = 0.04

# =====================================================================
# 8. SEGURANCA
# =====================================================================
MAX_FRAMES_PERDIDO = 12

# Durante/logo apos um desvio de cone a visao de pista perde a leitura por um
# instante com frequencia (a continua sai do campo, so ha traco da pontilhada).
# Decair o esterco e parar em 12 quadros (0.4 s) derrubava o carro bem no meio
# da manobra. Nesse periodo: segura o ultimo esterco, anda numa velocidade
# segura, e tolera ~2 s sem leitura.
MAX_FRAMES_PERDIDO_DESVIO   = 60
VELOCIDADE_PERDIDO_DESVIO   = 0.20
TURN_DECAI_PERDIDO_DESVIO   = 0.985

# =====================================================================
# 9. VIES DE FAIXA - andar mais perto da faixa PONTILHADA que do centro
# =====================================================================
LANE_BIAS_FRAC   = 0.85     # subiu de 0.70 -- bem mais perto do pontilhado
LANE_BIAS_LIMITE = 0.90     # subiu de 0.80 -- abriu mais espaco pro vies.
                            # ATENCAO: isso ja come bastante da margem de
                            # seguranca. Perto do limite, qualquer falha
                            # momentanea de deteccao do pontilhado vira erro
                            # grande rapido, porque o alvo ja esta quase
                            # colado nele.

# De que lado fica a faixa CONTINUA na sua pista: "direita", "esquerda" ou
# None (aprende sozinho olhando qual lado tem buracos).
# Fixo e MUITO mais seguro agora que o carro cruza a pontilhada pra desviar:
# no meio da manobra a linha externa do outro lado (tambem continua) parecia
# "a continua" pra visao, ela travava o lado errado e puxava o carro pra
# outra faixa. Se um dia rodar o percurso no sentido contrario, troque.
LADO_CONTINUO_FIXO = "direita"

# Faixa OPOSTA: depois de cruzar a pontilhada pra desviar, a camera de pista
# ve [linha externa do outro lado] + [pontilhada]. Essa dupla tem a MESMA
# largura de uma faixa, entao a visao a tomava por "a faixa" e puxava o carro
# pro centro da outra faixa. Um centro de faixa nao pode pular uma faixa
# inteira de um quadro pro outro: durante/logo apos um desvio, escolhe-se o
# centro (bruto ou +-2*hw) mais coerente com o quadro anterior. No uso normal
# (sem desvio) isso nunca liga, entao nao muda o comportamento de antes.
FAIXA_OPOSTA_JANELA = 90       # quadros (~3 s) apos o desvio em que ainda vale
FAIXA_OPOSTA_GATE   = 0.9      # o centro corrigido tem que cair a menos de
                               # GATE*hw do centro do quadro anterior

LADO_CONTINUO_SUAVIZACAO = 0.93
LADO_CONTINUO_MARGEM     = 0.15

# TRAVA da decisao de qual lado e continuo. Sem isso, um trecho ruim no meio
# do percurso (esterco violento perdendo o rastreio de um lado, camera
# engasgando) podia derrubar a taxa de deteccao daquele lado por um motivo
# que nao tem nada a ver com o padrao real da faixa, e o codigo TROCAVA de
# decisao no meio do percurso -- o alvo pulava pro lado oposto da pista.
# Agora, uma vez confiante (folga maior que a margem normal, sustentada por
# varios frames seguidos), a decisao trava de vez e para de reagir a
# chacoalhao no resto da execucao.
LADO_CONTINUO_TRAVA_MARGEM = 0.25   # folga maior que LADO_CONTINUO_MARGEM
                                     # pra comecar a contar pra travar
LADO_CONTINUO_TRAVA_FRAMES = 40     # frames CONSISTENTES seguidos pra travar

SAT_LIMITE = 0.95
MAX_FRAMES_SATURADO = 5


# =====================================================================
# 11. CONES (segunda camera, mais reta, so pra ver obstaculos)
# A camera da PISTA continua mandando em tudo: o carro segue a faixa igual
# antes. O cone so ADICIONA um deslocamento ao erro (pra desviar) e e
# sempre limitado pelas bordas da pista que a camera de pista enxerga.
#
# Pra descobrir os indices das cameras no Pi:
#     v4l2-ctl --list-devices
# Cada webcam USB cria 2 nos (video0/video1, video2/video3...). Use o
# PRIMEIRO de cada uma. Se o desvio ficar do lado errado, e so trocar os
# dois IDs aqui.
# =====================================================================
CONE_CAMERA_ATIVA = True
CAMERA_PISTA_ID   = 0
CAMERA_CONE_ID    = 2
CAMERA_CONE_RES   = (320, 240)   # cone e grande: nao precisa de resolucao alta
CAMERA_CONE_FPS   = 15           # menos banda USB e menos CPU pra decodificar MJPG;
                                 # a camera da pista fica com o FPS que precisa

# ---- deteccao por cor (HSV do OpenCV: H 0..179) ----
# Vermelho/laranja "dobra" o matiz: fica perto de 0 E perto de 179.
CONE_H_BAIXO    = 12      # aceita H de 0 ate aqui (vermelho -> laranja)
CONE_H_ALTO     = 168     # e de aqui ate 179
CONE_S_MIN      = 110     # saturacao minima (tira branco, cinza, parede)
CONE_V_MIN      = 70      # brilho minimo (tira sombra escura)
CONE_ROI_TOP_PCT = 0.10   # ignora a faixa de cima da imagem
CONE_AREA_MIN_PCT = 0.0015   # area minima do cone, fracao do frame
CONE_ASPECTO_REAL = 1.5   # altura/largura do cone inteiro (com a base). Usado
                          # pra estimar a largura quando a base sai do quadro.
CONE_ASPECTO_MIN = 0.60   # altura/largura minima do retangulo do cone
CONE_ASPECTO_MAX = 4.00
CONE_PREENCHIMENTO_MIN = 0.35   # area / area do retangulo (triangulo ~0.5)

# ---- distancia aproximada: altura do cone / altura da imagem ----
# Use Test_cones.py com o cone na distancia onde quer COMECAR a desviar
# (LONGE) e na distancia onde ja deve estar desviando por completo (PERTO).
CONE_ALTURA_LONGE = 0.04   # proximidade 0.0 (nao reage ainda)
CONE_ALTURA_PERTO = 0.09   # proximidade 1.0 (desvio total). Baixo de
                           # proposito: a folga ja cresce sozinha com o cone
                           # (larguras de cone), nao precisa de rampa lenta.

# ---- desvio ----
CONE_CORREDOR     = 0.95   # so reage a cone com |x| menor que isso (-1..+1)
# Folga lateral CENTRO DO CARRO -> CENTRO DO CONE, em cm da pista (MEDIR):
#   metade da largura do carro + metade da largura do cone + margem
# A margem cobre o erro de controle e de medida (no simulador o carro
# raspava com margem de ~4 cm e passava com ~8 cm). Raspou no cone -> suba
# CONE_MARGEM_CM. Desviando fundo demais pra outra faixa -> desca.
CARRO_LARGURA_CM = 20.0    # largura do carro (rodas externas)
CONE_MARGEM_CM   = 11.0

# Ganho do esterco do desvio. A lei e "pure pursuit": a curvatura necessaria
# pra passar por um ponto ao lado do cone cresce com 1/distancia^2, entao o
# mesmo erro de direcao exige MUITO mais volante com o cone perto. A largura
# do cone na imagem e proporcional a 1/distancia, entao ganho ~ K * largura.
# Nunca menor que PID_KP. Raspou/bateu com o cone perto -> suba.
CONE_ZONA_CENTRO  = 0.06   # |x| menor que isso = "bem no meio": usa o lado padrao
CONE_LADO_PADRAO  = "esquerda"  # por onde passa quando nao da pra saber
                                # (a pontilhada fica a esquerda na sua pista)

# ---- de que lado DA PONTILHADA o cone esta (MEDIR NA PISTA REAL) ----
# Regra: cone DENTRO da faixa do carro (entre a pontilhada e a continua) ->
# o carro cruza a pontilhada. Cone na outra faixa -> o carro se afasta dele,
# pro lado da continua.
# Como se sabe: na imagem da camera de cone, mede-se a distancia do cone ate
# a faixa CONTINUA (em larguras de cone, na altura da base do cone). Se ela e
# menor que a largura da faixa, o cone esta dentro da faixa do carro.
CONE_LARGURA_CM  = 18.0    # largura da BASE do cone no chao (meça!)
PISTA_LARGURA_CM = 29.0    # MEDIDO na pista: cone em cima da pontilhada -> a
                           # janela CONE leu 'continua a 29cm'. (Em cm so
                           # se o cone tiver mesmo CONE_LARGURA_CM; o que
                           # vale e a razao entre os dois.)
CONE_FOLGA_CM = CARRO_LARGURA_CM / 2.0 + CONE_LARGURA_CM / 2.0 + CONE_MARGEM_CM
CONE_FOLGA_LARGURAS = CONE_FOLGA_CM / CONE_LARGURA_CM    # (modo reserva)
CONE_NA_LINHA_CM = 5.0     # a menos disso da pontilhada = "em cima da linha":
                           # decide pelo lado com mais espaco
CONE_LINHA_TOPHAT_MIN = 15      # contraste minimo de uma linha na camera de cone
CONE_PALPITE_FORA_W = 0.3       # palpite (cone longe): o cone so conta como 'na outra
                                # faixa' se passar do centro da pista por mais que a
                                # meia-faixa + isso (em larguras de cone)
CONE_MEDIDA_ALTURA_MIN = 0.06   # a distancia ate a continua so vale com o cone pelo
                                # menos desse tamanho (medido no simulador: erro de
                                # 1-2 cm ja a partir de 0.08, ~2.5 m)
CONE_MEDIDA_ALTURA_MAX = 0.30   # acima disso (cone muito perto) a medida piora, em
                                # curva principalmente: so se usa se nao ha outra
CONE_CONF_GEO = 0.6             # peso do deslocamento quando so ha a estimativa
                                # geometrica (nunca se mediu a posicao do cone)
CONE_LINHA_PRESENCA   = 0.75    # fracao das linhas da faixa em que a linha
                                # tem que aparecer pra contar como CONTINUA
CONE_TROCA_PERSISTE_S = 0.45    # a regra tem que discordar por esse tempo
                                # pra o carro trocar de lado (medida ruidosa)
CONE_PISTA_DILATA_PX  = 21      # tira o cone da mascara da camera de PISTA
                                # (reflexos do plastico viravam "faixa"); 0 desliga
CONE_GANHO        = 1.0
CONE_ESTERCO_GANHO = 2.0
CONE_GANHO_POS    = 1.0   # modo 'posicao lateral': esterco = KP * isto * deslocamento
CONE_MARGEM_CONT_CM = 7.0 # o centro do carro fica a pelo menos isso da continua
CONE_OFFSET_NO_PID = False   # False: o deslocamento entra no ERRO do PID (D amortece)
                             # True : entra depois do PID (sem passar pelo D)
CONE_OFFSET_MAX   = 2.00   # teto do deslocamento somado ao erro. ALTO de
                           # proposito: com a pista puxando -0.8 e o cone
                           # exigindo +0.6, o desvio precisa ser 1.4. Quem
                           # limita de verdade sao as bordas da pista.
CONE_OFFSET_SUAVIZACAO = 0.5   # EMA do deslocamento
CONE_OFFSET_SLEW  = 0.8    # maximo que o deslocamento varia (erro/segundo). 1 unidade
                           # de erro ~ 29 cm de alvo na faixa: 1.5/s = ~40 cm/s, o
                           # que o carro consegue acompanhar (antes 6.0 era um degrau)
CONE_RETORNO_SLEW = 0.35   # idem, mas VOLTANDO pro centro da faixa (erro/segundo ~ 10 cm/s)
CONE_ERRO_SEM_SATURAR = True    # no desvio, soma o cone no erro da pista SEM o corte em +-1
CONE_NA_PISTA = False           # (experimental) mede o cone na camera da PISTA tambem; no simulador
                                # piorou, por isso desligado. Ligue so pra testar.
CONE_PISTA_YB_MIN = 0           # so mede com a base do cone abaixo desta fileira da imagem
CONE_PISTA_AREA_MIN = 40        # area minima (px) do vermelho na camera da pista
CONE_PISTA_EXTRAP_PX = 70       # quanto acima da zona varrida a reta do centro da faixa
                                # pode ser extrapolada pra achar a fileira do cone
CONE_PISTA_ASSOCIA_ALTURA = 0.10   # a medida da pista so vale com o cone frontal assim de perto
CONE_ACAO_ALTURA_SEM_MEDIDA = 0.31   # sem medida da posicao, espera o cone chegar mais perto
CONE_USA_POS_CARRO = True       # lado e posicao do cone = posicao REAL do carro (visao da
                                # pista) + lat do cone. Nao depende de ver a linha continua.
CONE_USA_FAIXA_CARRO = True     # decide o lado sabendo em QUAL faixa o carro esta
CONE_ACAO_ALTURA = 0.31        # cone no horizonte: mede mas so AGE quando passar disso
                               # (~1.3 m). Mais alto = espera chegar mais perto.
CONE_FRENTE_X = 0.5            # |x| menor que isso = o cone esta de frente (na faixa do carro)
CONE_FAIXA_TEMPO_S = 0.7       # tempo com o alvo na outra faixa pra considerar que o carro
                               # ja esta la
CONE_TRAVA_ALTURA_MIN = 0.12     # a trava so vale com o cone a vista e pelo menos assim de perto
CONE_TRAVA_LADO_CONE = False    # nunca esterca pro lado do cone que esta passando
CONE_PALPITE_VOLTA = True       # cone novo de frente logo apos cruzar: passa pelo lado de volta
CONE_COMPROMISSO_LONGE = 0.20   # um cone NOVO so e ignorado enquanto estiver mais longe que
                                # isso (~1 m); mais perto ele esta no caminho e e tratado
                                # (o log real mostrou o carro ignorando um cone ja de frente)
CONE_COMPROMISSO_ALTURA = 0.35   # cone assim de perto = o carro esta passando por ele;
                                # um cone novo nao derruba o desvio dele ate terminar

# A faixa PONTILHADA pode ser cruzada (e o lado da ultrapassagem); a CONTINUA
# nao. Com isso ligado, o limite do desvio do lado da pontilhada passa pra
# alem dela por CONE_CRUZAR_FRAC meias-larguras. Se desligar, o carro nunca
# sai da propria faixa (e em pista estreita pode nao conseguir passar).
CONE_PODE_CRUZAR_PONTILHADA = True
CONE_CRUZAR_FRAC = 0.8

# Rede de seguranca enquanto calibra: cone BEM na frente e quase encostando
# (altura relativa ou base cortada) -> para o carro em vez de bater.
# 0.0 desliga.
CONE_PARA_E_ANALISA = True      # PARA quando ve o cone, espera parar, analisa e so entao desvia (manobra.py)
CONE_GATILHO_ALTURA = 0.25      # altura do cone na imagem que dispara a parada. Cone de 25 cm:
                                # altura = 25*f/(dist*240), f~240px -> 0.21 ~ 120 cm, 0.25 ~ 100 cm, 0.28 ~ 90 cm, 0.50 ~ 50 cm.
                                # AJUSTE pelo 'alt' do log: e a distancia da parada.
CONE_GATILHO_QUADROS = 2        # quadros seguidos vendo o cone no gatilho (evita falso positivo)
CONE_PARADA_MIN_S = 0.3         # tempo minimo parado antes de analisar (inercia dos motores)
CONE_PARADA_MAX_S = 1.5         # se nao parar de se mexer nesse tempo, analisa assim mesmo
CONE_PARADA_CALMO_N = 2         # quadros seguidos 'sem se mexer' pra considerar parado
CONE_PARADO_D_ALT = 0.004       # o cone mudou de altura menos que isso entre quadros = parado
CONE_PARADO_D_X = 0.015         # idem, posicao lateral (-1..+1)
CONE_PARADO_DIFF = 3.0          # mudanca media (0-255) da imagem da pista = ainda mexendo
CONE_ANALISE_S = 0.25           # tempo medindo o cone com o carro parado
CONE_REARME_S = 0.5             # durante o desvio: cone sumiu por isso = ja passou; um novo pode parar
CONE_COOLDOWN_S = 1.0           # depois de um desvio, tempo sem parar de novo
CONE_OFFSET_SLEW_DESVIO = 3.0     # velocidade do desvio saindo da parada (erro/s); 0.8 = suave
CONE_DESVIO_MAX_S = 9.0         # seguranca: desvio nao dura mais que isso
CONE_GANHO_TURN_DESVIO = 1.0     # multiplica a direcao SO enquanto desvia (1.0 = igual; 1.5-2.0 = mais brusco)
CONE_FAIXA_PERSISTE = True      # depois de desviar FICA na outra faixa; so troca quando um cone na faixa dela pedir
CONE_PONT_TOL_CM = 14.0        # medida pela pontilhada que discorda da geometria por mais que isso e descartada
CONE_PARAR_ALTURA = 0.0    # parada DESLIGADA (o carro desvia em vez de parar)
                           # Se quiser a rede de seguranca: 0.50
CONE_TROCA_LADO_ALTURA_MAX = 0.25   # so troca de lado no meio da manobra se o
                                    # cone ainda esta mais longe que isso
CONE_PARAR_X      = 0.60   # so conta se o cone esta nesse corredor central
CONE_MARGEM_PISTA = 0.15   # fracao da meia-largura que NAO invade da borda

# Memoria: o cone some do campo de visao ANTES do carro terminar de passar
# por ele. Sem segurar o desvio, o carro volta pro centro e raspa o cone.
CONE_SEGURAR_S       = 1.60   # cone estava perto: mantem o desvio por esse tempo
                              # (tem que cobrir o carro INTEIRO passando)
CONE_SEGURAR_LONGE_S = 0.15   # cone estava longe: so tolera falha de deteccao
CONE_DECAI_S         = 0.30   # depois disso, volta ao normal gradualmente
CONE_RECONHECE_S     = 0.25   # sumiu por mais que isso -> proximo cone e outro

CONE_FREIO_MAX    = 0.10   # reduz ate 25% da velocidade com cone bem perto
CONE_FRAME_VELHO_S = 0.5   # frame da camera de cone mais velho que isso = ignora


# =====================================================================
# 10. COMO CALIBRAR O PID NO HARDWARE DE VERDADE (pra ganhar velocidade
# de verdade em curva, nao so via piso/freio)
#
# PASSO 1 -- so P (KI=0, KD=0)
#   python3 test_control.py (carro na mao, tracao desligada)
#   Suba PID_KP aos poucos ate o servo comecar a balancar sozinho.
#   Anote esse valor -- e o "Ku".
# PASSO 2 -- PID_KP = Ku * 0.5
# PASSO 3 -- sobe PID_KD aos poucos ate corrigir rapido sem passar do ponto
# PASSO 4 -- so se sobrar erro parado numa reta, sobe PID_KI devagar
#
# Um KP maior valido = correcao mais forte = pode manter mais velocidade
# em curva sem sair. E o proximo ganho real de velocidade depois deste.
# =====================================================================
