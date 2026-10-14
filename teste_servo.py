import RPi.GPIO as GPIO
import time

# ==============================
# CONFIGURAÇÃO DO SERVO
# ==============================

SERVO_PIN = 18

SERVO_CENTRO = 40
SERVO_MAX_ESQ = 0
SERVO_MAX_DIR = 90

# ==============================
# CONFIGURAÇÃO DO GPIO
# ==============================

GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)

# Frequência padrão para servo
servo = GPIO.PWM(SERVO_PIN, 50)
servo.start(0)


def mover_servo(angulo):
    """
    Move o servo para o ângulo informado.
    """

    # Converte o ângulo para Duty Cycle
    duty = 2.5 + (angulo / 18.0)

    servo.ChangeDutyCycle(duty)

    print(f"Servo -> {angulo}°")

    # Dá tempo para o servo chegar à posição
    time.sleep(1)

    # Para de enviar o pulso para evitar aquecimento/zumbido
    servo.ChangeDutyCycle(0)


try:
    print("Teste do servo iniciado!")

    # Esquerda - 0°
    mover_servo(SERVO_MAX_ESQ)

    # Centro - 40°
    mover_servo(SERVO_CENTRO)

    # Direita - 90°
    mover_servo(SERVO_MAX_DIR)

    # Volta para o centro
    mover_servo(SERVO_CENTRO)

    print("Teste finalizado!")

except KeyboardInterrupt:
    print("\nTeste interrompido pelo usuário.")

finally:
    servo.stop()
    GPIO.cleanup()
