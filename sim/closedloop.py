"""
sim/closedloop.py - SIMULADOR em malha fechada (sem carro, sem Raspberry).

Roda o codigo REAL do projeto (vision.py, cones.py, controller.py e a logica
do Main.py) contra: uma pista curva com pontilhada / continua / linha externa,
cones, um carro com dinamica (bicicleta + servo + latencia) e as DUAS cameras
renderizadas em perspectiva. Serve pra achar defeitos de controle/visao antes
de ir pra pista. Precisa de: pip install opencv-python numpy

Calibrado com a FOTO da pista real: faixas de ~1.5 larguras de cone, tres
linhas (externa esq. + pontilhada + continua dir.), pontilhado com tracos de
7.5 cm e vaos de 14 cm, cones um em cada faixa.

NAO substitui a pista real: as cameras do simulador (altura, angulo, campo de
visao) e a dinamica do carro sao estimativas. Use pra comparar versoes e achar
bugs, nao pra afinar numeros finais.

Uso rapido:  python sim/cenarios.py --pista S --semente 0 --mapa
"""
import sys, math, types
import os
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
import numpy as np
import cv2
import config
import controller
import cones as cones_mod
from vision import LaneVision
from controller import SteeringPID, speed_for
from piloto import Piloto


class Clock:
    t = 0.0


controller.time = types.SimpleNamespace(time=lambda: Clock.t)
cones_mod.time = types.SimpleNamespace(time=lambda: Clock.t)

# ------------------------------------------------------------ geometria
LANE_W = 0.29        # pontilhada -> continua (m)
CONE_W = 0.18        # base do cone (m)
CONE_H = 0.23
CAR_HALF_W = 0.10
CAR_REAR, CAR_FRONT = -0.06, 0.24     # em relacao ao eixo traseiro
WHEELBASE = 0.20
TAPE = 0.025


def catmull_closed(pts, n_per=120):
    P = np.array(pts, float); m = len(P); out = []
    for i in range(m):
        p0, p1, p2, p3 = P[(i - 1) % m], P[i], P[(i + 1) % m], P[(i + 2) % m]
        for t in np.linspace(0, 1, n_per, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t +
                              (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t +
                              (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out = np.array(out)
    seg = np.linalg.norm(np.diff(np.vstack([out, out[:1]]), axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    L = s[-1]
    ss = np.arange(0, L, 0.01)
    x = np.interp(ss, s, np.append(out[:, 0], out[0, 0]))
    y = np.interp(ss, s, np.append(out[:, 1], out[0, 1]))
    return np.stack([x, y], 1), L


class Track:
    def __init__(self, pts, outer_left_line=True):
        self.C, self.L = catmull_closed(pts)
        d = np.gradient(self.C, axis=0)
        self.T = d / np.linalg.norm(d, axis=1, keepdims=True)
        self.N = np.stack([self.T[:, 1], -self.T[:, 0]], 1)    # normal da DIREITA
        dT = np.gradient(self.T, axis=0) / 0.01
        self.kappa = (self.T[:, 0] * dT[:, 1] - self.T[:, 1] * dT[:, 0])
        self.n = len(self.C)
        self.outer_left = outer_left_line
        # quadrilateros das fitas: (K,4,2)
        self.quads = {}
        self.quads["dash"] = self._quads(0.0, dashed=True)
        self.quads["solid"] = self._quads(+LANE_W, dashed=False)
        if outer_left_line:
            self.quads["left"] = self._quads(-LANE_W, dashed=False)

    def _quads(self, off, dashed, step=0.04):
        q = []
        idx = np.arange(0, self.n - 4, int(step / 0.01))
        for k in idx:
            if dashed and ((k * 0.01) % 0.215) > 0.075:
                continue
            k2 = min(k + int(step / 0.01), self.n - 1)
            a = self.C[k] + off * self.N[k]; b = self.C[k2] + off * self.N[k2]
            na, nb = self.N[k], self.N[k2]
            h = TAPE / 2
            q.append([a - h * na, a + h * na, b + h * nb, b - h * nb])
        return np.array(q)

    def at(self, s):
        k = int(round(s / 0.01)) % self.n
        return self.C[k], self.T[k], self.N[k]

    def nearest(self, p, hint=None):
        if hint is None:
            d = np.linalg.norm(self.C - p, axis=1)
            return int(np.argmin(d))
        span = 120
        idx = (np.arange(hint - span, hint + span)) % self.n
        d = np.linalg.norm(self.C[idx] - p, axis=1)
        return int(idx[np.argmin(d)])


class Cam:
    def __init__(self, w, h, f, height, pitch_deg, fwd):
        self.w, self.h, self.f = w, h, f
        self.height = height
        self.pitch = math.radians(pitch_deg)
        self.fwd = fwd                 # a frente do eixo traseiro (m)

    def to_cam(self, pts, car):
        """pts (N,2) mundo -> (u, v, z)"""
        x, y, psi = car
        dx = (pts[:, 0] - x) * math.cos(psi) + (pts[:, 1] - y) * math.sin(psi) - self.fwd
        dy = -(pts[:, 0] - x) * math.sin(psi) + (pts[:, 1] - y) * math.cos(psi)
        cph, sph = math.cos(self.pitch), math.sin(self.pitch)
        z = dx * cph + self.height * sph
        yd = -dx * sph + self.height * cph
        xr = -dy
        zs = np.where(z > 1e-3, z, 1e-3)
        u = self.w / 2 + self.f * xr / zs
        v = self.h / 2 + self.f * yd / zs
        return u, v, z


def render(cam, track, cones, car, rng, floor=(38, 36, 36)):
    img = np.empty((cam.h, cam.w, 3), np.uint8)
    img[:] = floor
    # gradiente de luz + ruido de textura
    g = np.linspace(0.85, 1.15, cam.w)[None, :, None]
    img = np.clip(img * g, 0, 255).astype(np.uint8)
    img = cv2.add(img, rng.randint(0, 10, img.shape, dtype=np.uint8))

    # so quads perto do carro
    cx, cy, _ = car
    for name, Q in track.quads.items():
        if len(Q) == 0:
            continue
        mid = Q.mean(axis=1)
        near = np.where(np.linalg.norm(mid - np.array([cx, cy]), axis=1) < 3.5)[0]
        if near.size == 0:
            continue
        pts = Q[near].reshape(-1, 2)
        u, v, z = cam.to_cam(pts, car)
        u = u.reshape(-1, 4); v = v.reshape(-1, 4); z = z.reshape(-1, 4)
        ok = (z > 0.05).all(axis=1)
        for i in np.where(ok)[0]:
            poly = np.stack([u[i], v[i]], 1)
            if poly[:, 1].max() < 0 or poly[:, 1].min() > cam.h:
                continue
            if poly[:, 0].max() < -50 or poly[:, 0].min() > cam.w + 50:
                continue
            cv2.fillConvexPoly(img, np.round(poly).astype(np.int32), (225, 225, 225))

    # cones (ordem do mais longe pro mais perto)
    lst = []
    for (X, Y) in cones:
        u, v, z = cam.to_cam(np.array([[X, Y]]), car)
        if z[0] > 0.08:
            lst.append((z[0], u[0], v[0]))
    for z, u, v in sorted(lst, reverse=True):
        wpx = cam.f * CONE_W / z
        hpx = cam.f * CONE_H / z * math.cos(cam.pitch)
        if abs(u - cam.w / 2) > cam.w + wpx or v < -hpx or v > cam.h + 2 * hpx:
            continue
        pts = np.array([[u - wpx / 2, v], [u + wpx / 2, v],
                        [u + wpx * 0.14, v - hpx], [u - wpx * 0.14, v - hpx]], np.int32)
        cv2.fillConvexPoly(img, pts, (25, 35, 215))
        # reflexo branco do plastico (como o cone real)
        a = (int(u - wpx * 0.05), int(v - hpx * 0.75)); b = (int(u + wpx * 0.02), int(v - hpx * 0.25))
        cv2.rectangle(img, a, b, (205, 205, 235), -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


# ------------------------------------------------------------ carro
class Car:
    def __init__(self, x, y, psi, v_max=0.9):
        self.x, self.y, self.psi = x, y, psi
        self.delta = 0.0
        self.v_max = v_max
        self.v = 0.0

    def step(self, turn, duty, dt):
        # servo: turn -1..+1 -> roda ate 28 graus (+ = direita), com atraso 1a ordem
        alvo = turn * math.radians(28)
        self.delta += (alvo - self.delta) * min(1.0, dt / 0.10)
        v_alvo = self.v_max * duty
        self.v += (v_alvo - self.v) * min(1.0, dt / 0.15)
        self.x += self.v * math.cos(self.psi) * dt
        self.y += self.v * math.sin(self.psi) * dt
        self.psi -= self.v / WHEELBASE * math.tan(self.delta) * dt   # direita = horario

    @property
    def pose(self):
        return (self.x, self.y, self.psi)


def clearance(car, c):
    """distancia (m) da borda do carro ate a borda do cone (negativo = bateu)"""
    dx = (c[0] - car.x) * math.cos(car.psi) + (c[1] - car.y) * math.sin(car.psi)
    dy = -(c[0] - car.x) * math.sin(car.psi) + (c[1] - car.y) * math.cos(car.psi)
    qx = min(max(dx, CAR_REAR), CAR_FRONT)
    qy = min(max(dy, -CAR_HALF_W), CAR_HALF_W)
    return math.hypot(dx - qx, dy - qy) - CONE_W / 2


def hit_cone(car, c):
    dx = (c[0] - car.x) * math.cos(car.psi) + (c[1] - car.y) * math.sin(car.psi)
    dy = -(c[0] - car.x) * math.sin(car.psi) + (c[1] - car.y) * math.cos(car.psi)
    qx = min(max(dx, CAR_REAR), CAR_FRONT)
    qy = min(max(dy, -CAR_HALF_W), CAR_HALF_W)
    return math.hypot(dx - qx, dy - qy) < CONE_W / 2 * 0.9


# ------------------------------------------------------------ piloto
# O MESMO Piloto do carro (piloto.py): o simulador testa o codigo que anda na pista.
class Pilot(Piloto):
    dump_t = ()

    def __init__(self):
        super().__init__()
        self.offset = 0.0
        self._cone_img = None

    def step(self, lane_img, cone_img):
        self.debug_pista = any(abs(Clock.t - t) < 0.5 / 30 for t in self.dump_t)
        # a camera de cone entrega quadro novo a cada 2 quadros da pista; entre
        # eles o cone anterior continua valendo (como no carro)
        novo = cone_img is not None
        if novo:
            self._cone_img = cone_img
        turn, speed = super().step(Clock.t, lane_img, self._cone_img, novo)
        self.offset = self.desvio.offset
        if self.debug_pista and self.res.debug_view is not None:
            tag = f"dump_{Clock.t:05.2f}"
            cv2.imwrite(tag + "_lane.png", np.vstack([self.res.debug_view, self.res.debug_mask]))
        return turn, speed


LANE_CAM = Cam(480, 240, 215.0, 0.18, 30.0, 0.10)   # f=215: meia-largura da faixa ~135px no fundo (como a real)
CONE_CAM = Cam(320, 240, 250.0, 0.12, 0.0, 0.20)


def run(track, cones, s0=0.2, duty_scale=1.0, v_max=0.9, t_max=60.0, seed=0,
        verbose=False, start_u=0.145, log_window=None, cam_lane=LANE_CAM, cam_cone=CONE_CAM,
        stop_after_cones=True):
    rng = np.random.RandomState(seed)
    P, T, N = track.at(s0)
    p0 = P + start_u * N
    car = Car(p0[0], p0[1], math.atan2(T[1], T[0]), v_max=v_max)
    pilot = Pilot()
    Clock.t = 0.0
    dt = 1.0 / 30.0
    hits = set(); frames = 0
    clear = [9.0] * len(cones)
    hint = track.nearest(p0)
    hint0 = hint
    prog = 0.0
    prev_hint = hint
    max_u, min_u = -9, 9
    offtrack = 0
    log = []
    path = []
    lane_buf = []; cone_buf = []
    delay = 2        # frames de latencia
    cone_pos = [(c[0], c[1]) for c in cones]
    while Clock.t < t_max:
        lane_img = render(cam_lane, track, cone_pos, car.pose, rng)
        cone_img = render(cam_cone, track, cone_pos, car.pose, rng) if frames % 2 == 0 else None
        lane_buf.append(lane_img); cone_buf.append(cone_img)
        li = lane_buf[max(0, len(lane_buf) - 1 - delay)]
        ci = cone_buf[max(0, len(cone_buf) - 1 - delay)]
        turn, speed = pilot.step(li, ci)
        car.step(turn, speed, dt)
        Clock.t += dt; frames += 1

        # metricas
        pos = np.array([car.x, car.y])
        hint = track.nearest(pos, hint)
        dh = hint - prev_hint
        if dh > track.n / 2: dh -= track.n
        if dh < -track.n / 2: dh += track.n
        prog += dh * 0.01; prev_hint = hint
        rel = pos - track.C[hint]
        u = float(rel @ track.N[hint])
        max_u = max(max_u, u); min_u = min(min_u, u)
        if u > LANE_W + 0.10 or u < -LANE_W - 0.10:
            offtrack += 1
        for i, c in enumerate(cone_pos):
            cl = clearance(car, c)
            if cl < clear[i]:
                clear[i] = cl
            if i not in hits and hit_cone(car, c):
                hits.add(i)
        s_cur = hint * 0.01
        path.append((Clock.t, car.x, car.y, car.psi))
        if log_window is not None:
            log.append((Clock.t, s_cur, u, turn, speed, pilot.res.error, pilot.offset,
                        pilot.desvio.lado, pilot.desvio.na_faixa, pilot.desvio.dist_cont_cm, pilot.vision.faixa_oposta,
                        None if pilot.cone is None or not pilot.cone.found else
                        (pilot.cone.cx, pilot.cone.altura)))
        if frames > 30 and car.v < 0.01 and speed == 0.0 and not pilot.manobra.parado:
            break
        if prog >= track.L - 0.3:
            break
    return dict(hits=sorted(hits), offtrack=offtrack, max_u=max_u, min_u=min_u,
                t=Clock.t, s_end=prog, log=log, car=car, path=path, clear=clear, done=prog >= track.L - 0.3)
