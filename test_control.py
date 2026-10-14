"""
test_control.py - visao + PID + SERVO, com a TRACAO DESLIGADA.

Serve pra calibrar o PID (secao 10 do config.py) com o carro na mao, sem
risco de ele sair correndo. As rodas traseiras NAO giram.

    python3 test_control.py

Como usar (procedimento da secao 10 do config.py):
  1. Segure o carro sobre a pista, apontado pro centro dela.
     -> o servo tem que ficar RETO e o erro perto de 0.00
  2. Va subindo PID_KP com as teclas '+' / '-' (nao precisa editar o
     config.py e reiniciar toda vez).
  3. Com o carro parado numa posicao FIXA (nem centralizado, levemente
     torto), observe a coluna do angulo do servo ao longo do tempo.
     Se ele comecar a OSCILAR sozinho (o angulo fica subindo/descendo
     sem voce mexer no carro), esse e o Ku -- anota e aperta 'q'.
  4. Cole PID_KP = Ku * 0.5 no config.py e segue pro passo do KD.

Teclas: '+'/'-' ajusta PID_KP ao vivo | 'i' inverte o sentido | 'q' sai
"""

import time
import cv2

import config
from vision import LaneVision
from controller import SteeringPID
from MotorModule import Motor


def abrir_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_RES[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_RES[1])
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main():
    cap = abrir_camera()
    if not cap.isOpened():
        print("Camera nao abriu.")
        return

    vision = LaneVision()
    pid = SteeringPID()
    motor = Motor(config.ENA_A, config.IN1_A, config.IN2_A,
                  config.ENA_B, config.IN3_B, config.IN4_B,
                  config.SERVO_PIN)

    print("TRACAO DESLIGADA. So o servo se mexe.")
    print("Erro NEGATIVO -> servo deve ir pra DIREITA (corrigindo de volta)")
    print("Erro POSITIVO -> servo deve ir pra ESQUERDA")
    print("'+' sobe PID_KP | '-' desce | 'i' inverte | 'q' sai\n")

    frame_anterior_confiavel = True
    t_print = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.resize(frame, config.CAMERA_RES)
            res = vision.process(frame, debug=True)

            if res.valid and res.motivo == "ok" and (res.found_left or res.found_right):
                if not frame_anterior_confiavel:
                    pid.marcar_retomada()
                frame_anterior_confiavel = True
                turn = pid.update(res.error, res.heading)
            else:
                frame_anterior_confiavel = False
                turn = 0.0

            motor.turn_to(turn)

            # imprime em linha NOVA (nao sobrescreve) -- pra voce ver o
            # angulo do servo AO LONGO DO TEMPO e notar se ele comeca a
            # balancar sozinho com o carro parado.
            if time.time() - t_print >= 0.2:
                lado = "ESQUERDA" if turn < -0.05 else ("DIREITA" if turn > 0.05 else "RETO")
                ang = motor._last_angle if motor._last_angle is not None else config.SERVO_CENTRO
                print(f"erro {res.error:+.2f} | turn {turn:+.2f} | servo {ang:5.1f}deg "
                      f"-> {lado:8s} | KP {config.PID_KP:.2f} | inv {config.STEER_INVERT} "
                      f"| L{int(res.found_left)} R{int(res.found_right)}")
                t_print = time.time()

            cv2.imshow("Visao", res.debug_view)
            cv2.imshow("Mascara", res.debug_mask)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('i'):
                config.STEER_INVERT = not config.STEER_INVERT
                print(f"\n>> STEER_INVERT = {config.STEER_INVERT}\n")
            elif k in (ord('+'), ord('=')):
                config.PID_KP = round(min(3.0, config.PID_KP + 0.05), 2)
                print(f"\n>> PID_KP = {config.PID_KP:.2f}\n")
            elif k == ord('-'):
                config.PID_KP = round(max(0.05, config.PID_KP - 0.05), 2)
                print(f"\n>> PID_KP = {config.PID_KP:.2f}\n")

    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n\nCole no config.py:"
              f"\n  STEER_INVERT = {config.STEER_INVERT}"
              f"\n  PID_KP       = {config.PID_KP:.2f}"
              f"\n\n(Esse KP e o valor ATUAL, nao necessariamente o Ku onde"
              f"\n oscilou -- se ja passou do ponto de oscilar, use o valor"
              f"\n que voce anotou quando viu balancar, nao este.)\n")
        motor.cleanup()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
