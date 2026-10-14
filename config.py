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
VELOCIDADE_BASE    = 0.55   # BAIXA de proposito pra validar o desvio de cones.
                            # Suba de 0.05 em 0.05 (ate 1.00) conforme ficar
                            # estavel. NUNCA deixe abaixo de VELOCIDADE_MINIMA
                            # (0.45), senao o speed_for() inverte. O freio por
                            # FPS baixo e o de curva fechada continuam ativos.
VELOCIDADE_MINIMA  = 0.45   # subiu de 0.35 -- piso normal de curva
VELOCIDADE_PERDIDO = 0.22   # perdeu a pista, andando devagar procurando

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
SCAN_HW_MIN     = 25
SCAN_HW_MAX     = 160
SCAN_PESO_LONGE = 2.0
SCAN_TRACK_WINDOW = 95

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
VELOCIDADE_MINIMA_FECHADA = 0.38   # subiu de 0.32 -- piso mais alto so pra
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
CONE_ALTURA_LONGE = 0.05   # proximidade 0.0 (nao reage ainda)
CONE_ALTURA_PERTO = 0.15   # proximidade 1.0 (desvio total)

# ---- desvio ----
CONE_CORREDOR     = 0.95   # so reage a cone com |x| menor que isso (-1..+1)
# Folga lateral medida em LARGURAS DE CONE (nao em pixels da imagem!).
# x=0.45 na imagem e muito lado a 2m e quase nada a 40cm; largura de cone
# escala com a distancia. 1.6 largura ~ meia largura do carro + raio do cone
# + folga. Raspou no cone -> suba. Desviando demais -> desca.
CONE_FOLGA_LARGURAS = 1.6
CONE_ZONA_CENTRO  = 0.06   # |x| menor que isso = "bem no meio": usa o lado padrao
CONE_LADO_PADRAO  = "direita"   # por onde passa quando o cone esta centrado
CONE_GANHO        = 1.0
CONE_OFFSET_MAX   = 0.80   # maximo de deslocamento somado ao erro
CONE_OFFSET_SUAVIZACAO = 0.5   # EMA do deslocamento (evita chute do D no PID)
CONE_MARGEM_PISTA = 0.15   # fracao da meia-largura que NAO invade da borda

# Memoria: o cone some do campo de visao ANTES do carro terminar de passar
# por ele. Sem segurar o desvio, o carro volta pro centro e raspa o cone.
CONE_SEGURAR_S       = 1.20   # cone estava perto: mantem o desvio por esse tempo
                              # (tem que cobrir o carro INTEIRO passando)
CONE_SEGURAR_LONGE_S = 0.15   # cone estava longe: so tolera falha de deteccao
CONE_DECAI_S         = 0.80   # depois disso, volta ao normal gradualmente
CONE_RECONHECE_S     = 0.25   # sumiu por mais que isso -> proximo cone e outro
CONE_TROCA_CX        = 0.35   # salto de x entre frames -> trata como outro cone

CONE_FREIO_MAX    = 0.25   # reduz ate 25% da velocidade com cone bem perto
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
