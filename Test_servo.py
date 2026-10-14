"""
test_servo.py - acha o SERVO_CENTRO de verdade (roda reta).

    python3 test_servo.py

Teclas: 'a' -1 grau | 'd' +1 grau | 'q' sai imprimindo o valor.
Olhe a roda: quando estiver reta, aperte 'q' e cole o numero no config.py.
Os motores de tracao NAO ligam neste teste.
"""

import sys
import termios
import tty

import RPi.GPIO as GPIO
import config

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(config.SERVO_PIN, GPIO.OUT)

pwm = GPIO.PWM(config.SERVO_PIN, 50)
pwm.start(0)


def ler_tecla():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def set_angle(a):
    pwm.ChangeDutyCycle((a / 18.0) + 2.5)


angulo = float(config.SERVO_CENTRO)
set_angle(angulo)
print(f"Angulo atual: {angulo:.0f}  (a=-1  d=+1  q=sair)")

try:
    while True:
        k = ler_tecla()
        if k == 'a':
            angulo = max(config.SERVO_MAX_ESQ, angulo - 1)
        elif k == 'd':
            angulo = min(config.SERVO_MAX_DIR, angulo + 1)
        elif k in ('q', '\x03'):
            break
        else:
            continue
        set_angle(angulo)
        print(f"Angulo atual: {angulo:.0f}   ", end='\r', flush=True)
finally:
    print(f"\n\nCole no config.py:\n  SERVO_CENTRO = {int(angulo)}\n")
    pwm.stop()
    GPIO.cleanup()
