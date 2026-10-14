"""
Ver_cameras.py - mostra as DUAS cameras com o que a visao ve.
Nao liga motor, nao toca em GPIO: e so pra olhar e calibrar.

    python3 Ver_cameras.py          # janelas na tela do proprio Raspberry
    python3 Ver_cameras.py --web    # no navegador de outro PC (http://<ip>:8000)

Precisa rodar no terminal da AREA DE TRABALHO do Pi (onde ha tela). Pelo
terminal remoto (so texto) nao ha onde abrir janela: use --web.

Janelas:
  PISTA : faixas, alvo (ponto verde), erro   | embaixo: mascara do branco
  CONE  : retangulo do cone, x, lat (w = larguras de cone), altura, prox
          | ao lado: mascara do vermelho/laranja

IMPORTANTE: feche o Main.py antes (uma camera nao abre em dois programas).
Teclas: 'q' ou Ctrl+C encerra.
"""

import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

import config
from camera import abrir_camera, CameraThread
from vision import LaneVision
from cones import ConeVision

QUADROS = ("pista", "pista_mask", "cone", "cone_mask")
_lock = threading.Lock()
_jpg = {}


def guardar(nome, img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if ok:
        with _lock:
            _jpg[nome] = buf.tobytes()


def pegar(nome):
    with _lock:
        return _jpg.get(nome)


def texto(img, linhas, cor=(0, 255, 255)):
    for i, t in enumerate(linhas):
        cv2.putText(img, t, (6, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, cor, 1, cv2.LINE_AA)
    return img


class Analise:
    """Roda a visao nos frames e devolve as imagens ja com HUD."""

    def __init__(self):
        self.vision = LaneVision()
        self.cones = ConeVision()
        self.idx_p = self.idx_c = -1
        self._fps = {"p": [0.0, 0, time.time()], "c": [0.0, 0, time.time()]}

    def _fps_tick(self, k):
        f = self._fps[k]
        f[1] += 1
        if time.time() - f[2] >= 1.0:
            f[0], f[1], f[2] = f[1] / (time.time() - f[2]), 0, time.time()
        return f[0]

    def pista(self, cam):
        """-> (view, mask) ou None se nao chegou frame novo."""
        f, i, _ = cam.read()
        if f is None or i == self.idx_p:
            return None
        self.idx_p = i
        f = cv2.resize(f, config.CAMERA_RES)
        r = self.vision.process(f, debug=True)
        fps = self._fps_tick("p")
        lado = r.lado_continuo or "?"
        cor = (0, 255, 255) if r.valid else (0, 0, 255)
        linhas = [f"PISTA  {fps:.0f}fps  erro {r.error:+.2f}  "
                  f"L{int(r.found_left)} R{int(r.found_right)}  "
                  f"pts {r.n_pontos}  continua={lado}"]
        if not r.valid:
            linhas.append(f"PERDIDO: {r.motivo}")
        return texto(r.debug_view, linhas, cor), r.debug_mask

    def cone(self, cam):
        f, i, _ = cam.read()
        if f is None or i == self.idx_c:
            return None
        self.idx_c = i
        r = self.cones.process(f, debug=True)
        fps = self._fps_tick("c")
        if r.found:
            linhas = [f"CONE  {fps:.0f}fps  x {r.cx:+.2f}  lat {r.lat:+.1f}w",
                      f"altura {r.altura:.2f}  prox {r.proximidade:.2f}"]
            cor = (0, 255, 0)
        else:
            linhas = [f"CONE  {fps:.0f}fps  nao visto  cand {r.n_candidatos}"]
            cor = (0, 0, 255)
        return texto(r.debug_view, linhas, cor), r.debug_mask


def processar(cam_pista, cam_cone, parar):
    """Modo --web: processa em segundo plano e guarda JPEGs."""
    a = Analise()
    while not parar.is_set():
        feito = False
        if cam_pista is not None:
            r = a.pista(cam_pista)
            if r is not None:
                guardar("pista", r[0]); guardar("pista_mask", r[1]); feito = True
        if cam_cone is not None:
            r = a.cone(cam_cone)
            if r is not None:
                guardar("cone", r[0]); guardar("cone_mask", r[1]); feito = True
        if not feito:
            time.sleep(0.01)


def mostrar_janelas(cam_pista, cam_cone):
    """Modo padrao: janelas OpenCV na tela do proprio Pi."""
    a = Analise()
    while True:
        if cam_pista is not None:
            r = a.pista(cam_pista)
            if r is not None:
                cv2.imshow("PISTA (cima: visao | baixo: mascara)",
                           np.vstack([r[0], r[1]]))
        if cam_cone is not None:
            r = a.cone(cam_cone)
            if r is not None:
                view, mask = r
                if mask.shape[0] != view.shape[0]:      # mascara e so o recorte
                    pad = np.zeros_like(view)
                    pad[:mask.shape[0]] = mask
                    mask = pad
                cv2.imshow("CONE (esq: visao | dir: mascara)",
                           np.hstack([view, mask]))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


PAGINA = """<!doctype html><html lang="pt-br"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cameras do carro</title>
<style>
body{margin:0;background:#111;color:#ddd;font-family:system-ui,sans-serif}
h1{font-size:15px;font-weight:600;margin:0;padding:10px 14px;background:#1b1b1b}
.g{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:8px;padding:8px}
figure{margin:0;background:#000;border:1px solid #333}
figcaption{font-size:12px;padding:4px 8px;color:#9aa}
img{width:100%;display:block;image-rendering:auto}
</style></head><body>
<h1>Cameras do carro (so visualizacao - motores desligados)</h1>
<div class="g">
<figure><figcaption>PISTA - faixas e alvo</figcaption><img src="/stream/pista"></figure>
<figure><figcaption>CONE - deteccao</figcaption><img src="/stream/cone"></figure>
<figure><figcaption>PISTA - mascara (branco)</figcaption><img src="/stream/pista_mask"></figure>
<figure><figcaption>CONE - mascara (vermelho/laranja)</figcaption><img src="/stream/cone_mask"></figure>
</div></body></html>""".encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):      # sem spam no terminal
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(PAGINA)))
            self.end_headers()
            self.wfile.write(PAGINA)
            return

        if self.path.startswith("/stream/"):
            nome = self.path.split("/stream/", 1)[1]
            if nome not in QUADROS:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            ultimo = None
            try:
                while True:
                    jpg = pegar(nome)
                    if jpg is not None and jpg is not ultimo:
                        ultimo = jpg
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         b"Content-Length: " + str(len(jpg)).encode()
                                         + b"\r\n\r\n" + jpg + b"\r\n")
                    time.sleep(0.04)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            return

        self.send_error(404)


def ip_local():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))     # nao envia nada, so descobre a rota
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "<ip-do-pi>"


def main():
    web = "--web" in sys.argv
    args = [x for x in sys.argv[1:] if x != "--web"]
    porta = int(args[0]) if args else 8000

    if not web and not os.environ.get("DISPLAY"):
        print("Nao ha tela neste terminal (DISPLAY vazio).")
        print("  -> rode isto no terminal da AREA DE TRABALHO do Raspberry, ou")
        print("  -> use:  python3 Ver_cameras.py --web")
        return

    cap_p = abrir_camera(config.CAMERA_PISTA_ID, config.CAMERA_RES, "pista")
    cam_pista = CameraThread(cap_p) if cap_p.isOpened() else None
    if cam_pista is None:
        print("AVISO: camera da pista nao abriu.")

    cap_c = abrir_camera(config.CAMERA_CONE_ID, config.CAMERA_CONE_RES, "cone",
                         fps=config.CAMERA_CONE_FPS)
    cam_cone = CameraThread(cap_c) if cap_c.isOpened() else None
    if cam_cone is None:
        print("AVISO: camera de cone nao abriu.")

    if cam_pista is None and cam_cone is None:
        print("Nenhuma camera abriu. Feche o Main.py e confira os IDs no config.py.")
        return

    try:
        if web:
            parar = threading.Event()
            threading.Thread(target=processar, args=(cam_pista, cam_cone, parar),
                             daemon=True).start()
            srv = ThreadingHTTPServer(("0.0.0.0", porta), Handler)
            srv.daemon_threads = True
            print(f"\nAbra no navegador do PC:  http://{ip_local()}:{porta}")
            print("(Ctrl+C encerra)\n")
            try:
                srv.serve_forever()
            finally:
                parar.set()
                srv.server_close()
        else:
            print("\nJanelas abertas na tela do Pi. Tecla 'q' (ou Ctrl+C) encerra.\n")
            mostrar_janelas(cam_pista, cam_cone)
    except KeyboardInterrupt:
        print("\nEncerrando.")
    finally:
        if cam_pista is not None:
            cam_pista.release()
        if cam_cone is not None:
            cam_cone.release()
        if not web:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
