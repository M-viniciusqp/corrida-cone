"""
registro.py - grava o que o carro fez em cada corrida, pra analisar DEPOIS.

Cria  logs/AAAAMMDD_HHMMSS/  com:
  eventos.txt     todas as linhas [manobra] / [desvio] com a hora
  telemetria.csv  ~10 linhas por segundo: estado, velocidade, direcao, erro, cone...
  fotos/          uma foto (pista + cone lado a lado) a cada decisao da manobra:
                  cone visto, parou, lado escolhido, cone ignorado, fim do desvio

Assim, quando o carro erra num ponto, da pra olhar a foto e a linha do evento
daquele ponto em vez de depender de print de tela. Desligar: python3 Main.py --sem-log
"""

import csv
import os
import time

import cv2
import numpy as np


class Registro:
    def __init__(self, base="logs"):
        self.t0 = time.time()
        nome = time.strftime("%Y%m%d_%H%M%S")
        self.pasta = os.path.join(base, nome)
        self.pasta_fotos = os.path.join(self.pasta, "fotos")
        os.makedirs(self.pasta_fotos, exist_ok=True)
        self._ev = open(os.path.join(self.pasta, "eventos.txt"), "w", buffering=1,
                        encoding="utf-8")
        self._csv_f = open(os.path.join(self.pasta, "telemetria.csv"), "w", newline="",
                           buffering=1, encoding="utf-8")
        self._csv = csv.writer(self._csv_f)
        self._csv.writerow(["t", "estado", "fps", "speed", "turn", "erro", "erro_u", "pts",
                            "lado", "cone_visto", "cone_x", "cone_alt", "cone_lat_w",
                            "w_cone_cm", "w_alvo_cm", "fonte", "offset", "faixa_outra"])
        self._t_csv = 0.0
        self._n_foto = 0
        print(f"Gravando log em {self.pasta}")

    def _t(self):
        return time.time() - self.t0

    def evento(self, texto):
        self._ev.write(f"[{self._t():7.2f}s] {texto}\n")

    def telemetria(self, piloto, fps):
        agora = time.time()
        if agora - self._t_csv < 0.1:
            return
        self._t_csv = agora
        r, c, d = piloto.res, piloto.cone, piloto.desvio
        visto = c is not None and c.found
        self._csv.writerow([
            f"{self._t():.2f}", piloto.manobra.estado, f"{fps:.1f}",
            f"{piloto.speed:.2f}", f"{piloto.turn:+.2f}",
            f"{r.error:+.2f}", f"{r.error_u:+.2f}", r.n_pontos, d.lado,
            int(visto), f"{c.cx:+.2f}" if visto else "", f"{c.altura:.2f}" if visto else "",
            f"{c.lat:+.1f}" if visto else "",
            f"{d.w_cone_cm:.0f}" if d.w_cone_cm is not None else "",
            f"{d.w_alvo_cm:.0f}" if d.w_alvo_cm is not None else "",
            d.fonte_w, f"{d.offset:+.2f}", int(d.faixa_outra)])

    def foto(self, tag, frame_pista, frame_cone, piloto):
        """Foto da decisao: camera da pista | camera do cone, com o estado escrito."""
        try:
            a = frame_pista.copy()
            if frame_cone is not None:
                b = frame_cone.copy()
                h = a.shape[0]
                b = cv2.resize(b, (int(b.shape[1] * h / b.shape[0]), h))
                c = piloto.cone
                if c is not None and c.found and c.bbox is not None:
                    sx = b.shape[1] / frame_cone.shape[1]
                    x, y, w, hh = c.bbox
                    cv2.rectangle(b, (int(x * sx), int(y * sx)), (int((x + w) * sx),
                                  int((y + hh) * sx)), (0, 255, 0), 2)
                img = np.hstack([a, b])
            else:
                img = a
            d = piloto.desvio
            linhas = [f"{tag}  lado {d.lado:+d}",
                      f"cone {d.w_cone_cm:+.0f}cm [{d.fonte_w}]" if d.w_cone_cm is not None
                      else "cone: posicao nao medida"]
            if piloto.manobra.ultimo:
                linhas.append(piloto.manobra.ultimo[:70])
            for i, t in enumerate(linhas):
                cv2.putText(img, t, (6, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 255, 255), 1, cv2.LINE_AA)
            self._n_foto += 1
            nome = f"{self._n_foto:03d}_{self._t():06.2f}s_{tag}.jpg"
            cv2.imwrite(os.path.join(self.pasta_fotos, nome), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 80])
        except Exception as e:           # log nunca derruba o carro
            self.evento(f"(falha ao salvar foto: {e})")

    def fechar(self):
        for f in (self._ev, self._csv_f):
            try:
                f.close()
            except Exception:
                pass
