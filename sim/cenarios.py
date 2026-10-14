"""
sim/cenarios.py - cenarios de teste do simulador (pista, cones, mapa, CLI).

    python sim/cenarios.py                       # semente 0, velocidade 0.8
    python sim/cenarios.py --semente 2 --mapa    # salva mapa_sementeN.png
    python sim/cenarios.py --todas               # sementes 0..4
    python sim/cenarios.py --sem-externa         # pista sem a linha externa
    python sim/cenarios.py --pista S --todas     # pista em S, cones alternados
"""
import argparse
import math
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.dirname(AQUI))
import numpy as np
import cv2
import config
from closedloop import Track, Cam, run, LANE_W, CONE_W

# oval com retas e curvas (raio minimo ~0.77 m)
PONTOS = [(0, 0), (2.0, -0.3), (4.0, 0.0), (4.6, 1.2), (3.8, 2.6),
          (2.0, 3.0), (0.2, 2.6), (-0.7, 1.4)]


def pista(escala=1.0, externa=True):
    return Track([(x * escala, y * escala) for x, y in PONTOS], outer_left_line=externa)


def cones_aleatorios(tr, n, semente, espaco=1.9):
    """lista de (s, u, (x, y)); u = cm/100 a partir da pontilhada (+ = direita)"""
    rng = np.random.RandomState(semente)
    out, s = [], 2.2
    while len(out) < n and s < tr.L - 1.5:
        u = rng.uniform(-0.22, -0.02) if rng.rand() < 0.2 else rng.uniform(0.02, 0.27)
        P, T, N = tr.at(s)
        out.append((s, u, tuple(P + u * N)))
        s += espaco + rng.uniform(0.0, 0.8)
    return out


def pista_S(amp=0.20, comp=3.0, externa=True):
    """oval com curvas em S somadas (como a pista real da foto).
    amp 0.2 e moderada; 0.3 ja e dificil ate pro seguidor sem cones."""
    base = pista(1.0, externa=False)
    n_ond = max(1, round(base.L / comp))
    pts, s = [], 0.0
    while s < base.L - 0.15:
        P, T, N = base.at(s)
        pts.append(tuple(P + amp * math.sin(2 * math.pi * s / (base.L / n_ond)) * N))
        s += 0.18
    return Track(pts, outer_left_line=externa)


def cones_alternados(tr, n, espaco, primeiro="R", u_r=0.145, u_l=-0.145, s0=2.5):
    """cones um em cada faixa (como na foto): direita, esquerda, direita..."""
    out = []
    for i in range(n):
        s = s0 + i * espaco
        if s > tr.L - 1.5:
            break
        lado = primeiro if i % 2 == 0 else ("L" if primeiro == "R" else "R")
        u = u_r if lado == "R" else u_l
        P, T, N = tr.at(s)
        out.append((s, u, tuple(P + u * N)))
    return out


def desenhar_mapa(tr, cones, out, nome, px_m=110, margem=0.6):
    C = tr.C
    allp = np.vstack([C, np.array(cones)]) if cones else C
    x0, y0 = allp.min(0) - margem
    x1, y1 = allp.max(0) + margem
    W, H = int((x1 - x0) * px_m), int((y1 - y0) * px_m)
    img = np.full((H, W, 3), 30, np.uint8)

    def P(p):
        return (int((p[0] - x0) * px_m), int(H - (p[1] - y0) * px_m))

    for off, col in ((0.0, (150, 150, 150)), (LANE_W, (235, 235, 235)), (-LANE_W, (235, 235, 235))):
        pts = np.array([P(c + off * n) for c, n in zip(C[::4], tr.N[::4])], np.int32)
        if off == 0.0:
            for i in range(0, len(pts) - 1, 4):
                cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), col, 1)
        else:
            cv2.polylines(img, [pts], True, col, 1)
    for i, c in enumerate(cones):
        cv2.circle(img, P(c), int(CONE_W / 2 * px_m), (40, 40, 220), -1)
        cv2.putText(img, str(i), (P(c)[0] + 8, P(c)[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    for k, (t, x, y, psi) in enumerate(out["path"]):
        cv2.circle(img, P((x, y)), 1, (0, 255, 0), -1)
        if k % 60 == 0:
            cv2.putText(img, f"{t:.0f}s", (P((x, y))[0] + 4, P((x, y))[1] + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    cv2.imwrite(nome, img)
    print("mapa salvo em", nome)


def rodar(semente, vmax, externa=True, escala=1.0, mapa=False, tipo="oval", espaco=2.0):
    if tipo == "S":
        tr = pista_S(externa=externa)
        cs = cones_alternados(tr, 6, espaco, primeiro="R" if semente % 2 == 0 else "L")
    else:
        tr = pista(escala, externa)
        cs = cones_aleatorios(tr, 7, semente)
    pos = [c[2] for c in cs]
    out = run(tr, pos, v_max=vmax, t_max=tr.L / (vmax * 0.35) * 1.3)
    cl = " ".join(f"{c * 100:+3.0f}" if c < 5 else "  -" for c in out["clear"])
    print(f"semente {semente} vmax {vmax}: volta={'OK ' if out['done'] else 'NAO'} "
          f"batidas={len(out['hits'])}/{len(cs)} fora_da_pista={out['offtrack']:3d} "
          f"progresso={out['s_end']:.1f}/{tr.L:.1f} m | folga por cone (cm): {cl}")
    if mapa:
        desenhar_mapa(tr, pos, out, f"mapa_semente{semente}.png")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--vmax", type=float, default=0.8, help="velocidade maxima do carro simulado (m/s a duty 1.0)")
    ap.add_argument("--todas", action="store_true")
    ap.add_argument("--sem-externa", action="store_true")
    ap.add_argument("--mapa", action="store_true")
    ap.add_argument("--pista", choices=["oval", "S"], default="oval",
                    help="S = curvas em S + cones alternando faixa direita/esquerda")
    ap.add_argument("--espaco", type=float, default=2.0, help="metros entre cones (pista S)")
    a = ap.parse_args()
    for s in (range(5) if a.todas else [a.semente]):
        rodar(s, a.vmax, externa=not a.sem_externa, mapa=a.mapa, tipo=a.pista, espaco=a.espaco)
