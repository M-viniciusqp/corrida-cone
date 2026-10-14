"""
Ver_cameras.py - mostra as DUAS cameras no navegador, com o que a visao ve.
Nao liga motor, nao toca em GPIO: e so pra olhar e calibrar.

    python3 Ver_cameras.py          # porta 8000
    python3 Ver_cameras.py 8080     # outra porta

Depois abra no PC (na MESMA rede do Pi):

    http://<ip-do-pi>:8000          (o IP aparece quando o script inicia)

Quadros mostrados:
  - PISTA: faixas detectadas, alvo (ponto verde) e o erro
  - PISTA mascara: o que a visao considera "branco"
  - CONE: retangulo do cone, x, distancia lateral (w = larguras de cone)
  - CONE mascara: o que a visao considera "vermelho/laranja"

IMPORTANTE: feche o Main.py antes (uma camera nao abre em dois programas).
Ctrl+C encerra.
"""

import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

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


def processar(cam_pista, cam_cone, parar):
    vision = LaneVision()
    cones = ConeVision()
    idx_p = idx_c = -1
    fps_p = fps_c = 0.0
    t_p = t_c = time.time()
    n_p = n_c = 0

    while not parar.is_set():
        feito = False

        if cam_pista is not None:
            f, i, _ = cam_pista.read()
            if f is not None and i != idx_p:
                idx_p = i
                f = cv2.resize(f, config.CAMERA_RES)
                r = vision.process(f, debug=True)
                n_p += 1
                if time.time() - t_p >= 1.0:
                    fps_p, n_p, t_p = n_p / (time.time() - t_p), 0, time.time()
                lado = r.lado_continuo or "?"
                cor = (0, 255, 255) if r.valid else (0, 0, 255)
                linhas = [f"PISTA  {fps_p:.0f}fps  erro {r.error:+.2f}  "
                          f"L{int(r.found_left)} R{int(r.found_right)}  "
                          f"pts {r.n_pontos}  continua={lado}"]
                if not r.valid:
                    linhas.append(f"PERDIDO: {r.motivo}")
                guardar("pista", texto(r.debug_view, linhas, cor))
                guardar("pista_mask", r.debug_mask)
                feito = True

        if cam_cone is not None:
            f, i, _ = cam_cone.read()
            if f is not None and i != idx_c:
                idx_c = i
                r = cones.process(f, debug=True)
                n_c += 1
                if time.time() - t_c >= 1.0:
                    fps_c, n_c, t_c = n_c / (time.time() - t_c), 0, time.time()
                if r.found:
                    linhas = [f"CONE  {fps_c:.0f}fps  x {r.cx:+.2f}  lat {r.lat:+.1f}w",
                              f"altura {r.altura:.2f}  prox {r.proximidade:.2f}"]
                    cor = (0, 255, 0)
                else:
                    linhas = [f"CONE  {fps_c:.0f}fps  nao visto  cand {r.n_candidatos}"]
                    cor = (0, 0, 255)
                guardar("cone", texto(r.debug_view, linhas, cor))
                guardar("cone_mask", r.debug_mask)
                feito = True

        if not feito:
            time.sleep(0.01)


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
    porta = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

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

    parar = threading.Event()
    threading.Thread(target=processar, args=(cam_pista, cam_cone, parar),
                     daemon=True).start()

    srv = ThreadingHTTPServer(("0.0.0.0", porta), Handler)
    srv.daemon_threads = True
    print(f"\nAbra no navegador do PC:  http://{ip_local()}:{porta}\n"
          f"(Ctrl+C encerra)\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrando.")
    finally:
        parar.set()
        srv.server_close()
        if cam_pista is not None:
            cam_pista.release()
        if cam_cone is not None:
            cam_cone.release()


if __name__ == '__main__':
    main()
