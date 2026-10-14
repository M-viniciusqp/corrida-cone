"""Controle de direcao (PID) desacoplado da visao e dos motores."""

import time
import config


class SteeringPID:
    def __init__(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_t = time.time()
        self.first = True
        # ultimos termos calculados, so pra diagnostico/telemetria -- nao
        # influenciam o controle, sao so pra dar visibilidade de qual termo
        # esta dominando o comando final
        self.last_p = 0.0
        self.last_i = 0.0
        self.last_d = 0.0
        self.last_ff = 0.0
        self.smooth_derivative = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_t = time.time()
        self.first = True
        self.smooth_derivative = 0.0

    def marcar_retomada(self):
        """Chame isso quando o frame anterior estava com leitura congelada/
        degradada e este frame volta a ter leitura fresca e confiavel.

        Pula o termo D UMA vez (igual ja fazemos no primeiro frame). Sem
        isso, o D compara o erro fresco de agora com o erro CONGELADO de
        antes, ve um salto que nao e curva de verdade, e dispara um esterco
        artificial. Nao mexe no integral -- so evita o chute do D nessa
        transicao especifica.
        """
        self.first = True
        self.prev_t = time.time()
        self.smooth_derivative = 0.0

    def update(self, error, heading=0.0):
        now = time.time()
        dt = now - self.prev_t
        if dt <= 0 or dt > 0.5:      # protege contra travada de frame
            dt = 1 / 30.0
        self.prev_t = now

        if abs(error) < config.DEAD_ZONE:
            error = 0.0
            self.integral *= 0.5     # sangra o integral no reto

        self.integral += error * dt
        self.integral = max(-config.PID_I_LIMIT,
                            min(config.PID_I_LIMIT, self.integral))

        # no primeiro frame nao existe derivada valida -> evita um chute inicial
        raw_derivative = 0.0 if self.first else (error - self.prev_error) / dt
        raw_derivative = max(-5.0, min(5.0, raw_derivative))   # trava contra spike de frame travado
        self.prev_error = error
        self.first = False

        # FILTRO da derivada. O FPS varia (14 a 25 no seu log), entao o dt
        # varia frame a frame -- derivada = delta_erro/dt amplifica muito
        # qualquer ruido pequeno quando o dt e curto. Sem filtro, isso gera
        # picos que chegam a INVERTER o sinal do comando final mesmo com o
        # erro quase parado (foi exatamente o que apareceu no log real:
        # erro -0.36 mas turn +0.17 por causa do D sozinho). Suaviza igual
        # ja fazemos com o erro e o heading.
        b = config.PID_D_SUAVIZACAO
        self.smooth_derivative = b * self.smooth_derivative + (1 - b) * raw_derivative
        derivative = self.smooth_derivative

        turn = (config.PID_KP * error +
                config.PID_KI * self.integral +
                config.PID_KD * derivative)

        # feed-forward: se a pista ja esta inclinando, antecipa a curva
        # ANTES do erro lateral crescer. E o que evita chegar atrasado numa
        # curva fechada.
        ff = config.FEEDFORWARD_HEADING * heading
        turn += ff

        # guarda os termos crus (antes de inverter/grampear) pra telemetria
        self.last_p = config.PID_KP * error
        self.last_i = config.PID_KI * self.integral
        self.last_d = config.PID_KD * derivative
        self.last_ff = ff

        if config.STEER_INVERT:
            turn = -turn

        return max(-1.0, min(1.0, turn))


def speed_for(turn, curvature):
    """Reduz a velocidade proporcional a agressividade da curva.

    Tem DOIS pisos, nao um so:
    - VELOCIDADE_MINIMA: piso normal, vale pra qualquer curva.
    - VELOCIDADE_MINIMA_FECHADA: piso mais baixo, so entra em acao quando a
      curvatura passa de CURVA_FECHADA_LIMIAR.

    Por que dois: o raio de giro do carro (servo no talo) tem um limite
    fisico. Numa curva MUITO fechada, mesmo com o erro lido certo e o
    esterco no maximo, se a velocidade for alta demais o carro simplesmente
    nao vira rapido o suficiente -- sai da pista mesmo com o controle
    correto. Isso nao e bug de PID, e fisica. A unica correcao e reduzir
    a velocidade so NESSA curva especifica, sem penalizar as curvas suaves
    que ja estao rapidas e funcionando bem.
    """
    fator = 1.0 - min(1.0, abs(turn) * config.SPEED_TURN_PESO +
                      curvature * config.SPEED_CURVATURA_PESO)
    v = config.VELOCIDADE_MINIMA + (
        config.VELOCIDADE_BASE - config.VELOCIDADE_MINIMA) * max(0.0, fator)

    # freio extra SO pra curva fechada de verdade (transicao suave, nao abrupta)
    if curvature > config.CURVA_FECHADA_LIMIAR:
        excesso = min(1.0, (curvature - config.CURVA_FECHADA_LIMIAR) /
                      config.CURVA_FECHADA_RANGE)
        v -= excesso * (config.VELOCIDADE_MINIMA - config.VELOCIDADE_MINIMA_FECHADA)
        v = max(v, config.VELOCIDADE_MINIMA_FECHADA)

    return v
