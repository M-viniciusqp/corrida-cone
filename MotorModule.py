import RPi.GPIO as GPIO
from time import sleep
import config

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)


class Motor:
    """Tracao traseira (ponte H dupla) + direcao por servo."""

    def __init__(self, EnaA, In1A, In2A, EnaB, In1B, In2B, servoPin):
        self.EnaA, self.In1A, self.In2A = EnaA, In1A, In2A
        self.EnaB, self.In1B, self.In2B = EnaB, In1B, In2B

        for pin in (EnaA, In1A, In2A, EnaB, In1B, In2B, servoPin):
            GPIO.setup(pin, GPIO.OUT)

        self.pwmA = GPIO.PWM(self.EnaA, 100)
        self.pwmB = GPIO.PWM(self.EnaB, 100)
        self.pwmA.start(0)
        self.pwmB.start(0)

        self.servoPin = servoPin
        self.pwmServo = GPIO.PWM(self.servoPin, 50)
        self.pwmServo.start(0)

        self._last_angle = None
        self.set_servo_angle(config.SERVO_CENTRO)

    # ---------------- SERVO ----------------
    def set_servo_angle(self, angle):
        angle = max(config.SERVO_MAX_ESQ, min(config.SERVO_MAX_DIR, angle))
        # so reescreve o duty se mudou de verdade -> mata o tremor do servo
        if self._last_angle is not None and \
           abs(angle - self._last_angle) < config.SERVO_DEADBAND_DEG:
            return
        self._last_angle = angle
        duty = (angle / 18.0) + 2.5
        self.pwmServo.ChangeDutyCycle(duty)

    def turn_to(self, turn):
        """turn em -1 (esquerda total) .. +1 (direita total)."""
        turn = max(-1.0, min(1.0, turn))
        if turn < 0:
            amplitude = config.SERVO_CENTRO - config.SERVO_MAX_ESQ
        else:
            amplitude = config.SERVO_MAX_DIR - config.SERVO_CENTRO
        self.set_servo_angle(config.SERVO_CENTRO + turn * amplitude)

    # ---------------- TRACAO ----------------
    def drive(self, speed):
        """speed em -1 .. +1."""
        speed = max(-1.0, min(1.0, speed)) * 100
        forward = speed >= 0
        GPIO.output(self.In1A, GPIO.HIGH if forward else GPIO.LOW)
        GPIO.output(self.In2A, GPIO.LOW if forward else GPIO.HIGH)
        GPIO.output(self.In1B, GPIO.HIGH if forward else GPIO.LOW)
        GPIO.output(self.In2B, GPIO.LOW if forward else GPIO.HIGH)
        self.pwmA.ChangeDutyCycle(abs(speed))
        self.pwmB.ChangeDutyCycle(abs(speed))

    def move(self, speed=0.5, turn=0.0, t=0):
        self.turn_to(turn)
        self.drive(speed)
        if t:
            sleep(t)

    def stop(self, t=0):
        self.pwmA.ChangeDutyCycle(0)
        self.pwmB.ChangeDutyCycle(0)
        self.set_servo_angle(config.SERVO_CENTRO)
        if t:
            sleep(t)

    def cleanup(self):
        self.stop()
        sleep(0.2)
        self.pwmA.stop(); self.pwmB.stop(); self.pwmServo.stop()
        GPIO.cleanup()
