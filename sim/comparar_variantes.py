"""Compara variantes de config no simulador, em paralelo.

Uso:  python sim/comparar_variantes.py final,old S 6
      python sim/comparar_variantes.py final oval 10 CONE_ACAO_ALTURA=0.15
(variantes em VARIANTS; o 4o argumento sobrescreve valores do config)
"""
import sys, os, time, itertools, json
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "sim"))
from multiprocessing import Pool

VARIANTS = {
 "final":      dict(),                              # o config como esta
 "sem_manobra": dict(CONE_PARA_E_ANALISA=False),    # desvia andando, sem parar
 "persiste":   dict(CONE_FAIXA_PERSISTE=True),      # fica na faixa pra onde desviou
}

def job(args):
    var, tipo, sem, extra = args
    import config, cenarios, io, contextlib
    for k, v in VARIANTS[var].items(): setattr(config, k, v)
    for k, v in extra.items(): setattr(config, k, v)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        if tipo == "ovalalt":
            tr = cenarios.pista(1.0, True)
            cs = cenarios.cones_alternados(tr, 6, 2.2, primeiro="R" if sem % 2 == 0 else "L", s0=2.0 + 0.15 * sem)
            from closedloop import run
            o = run(tr, [c[2] for c in cs], v_max=0.8, t_max=tr.L / (0.8 * 0.35) * 1.3)
            print(f"semente {sem}: batidas={len(o['hits'])}/{len(cs)} progresso={o['s_end']:.1f}/{tr.L:.1f}")
        else:
            o = cenarios.rodar(sem, 0.8, tipo=tipo)
    return (var, tipo, sem, len(o["hits"]), o["done"], round(o["s_end"], 1), o["offtrack"])

if __name__ == "__main__":
    os.environ["SIM_VEL55"] = "1"
    variants = sys.argv[1].split(",")
    tipo = sys.argv[2]
    seeds = range(int(sys.argv[3]))
    import ast
    extra = {}
    if len(sys.argv) > 4 and sys.argv[4] != '-':
        for kv in sys.argv[4].split(','):
            k, v = kv.split('=')
            extra[k] = ast.literal_eval(v)
    jobs = [(v, tipo, s, extra) for v in variants for s in seeds]
    t = time.time()
    with Pool(max(1, os.cpu_count() - 1)) as p:
        res = p.map(job, jobs)
    for v in variants:
        rs = [r for r in res if r[0] == v]
        print(f"{v:5s} {tipo}: batidas={sum(r[3] for r in rs):2d}  voltas={sum(r[4] for r in rs)}/{len(rs)}  "
              f"progresso_medio={sum(r[5] for r in rs)/len(rs):.1f}  fora={sum(r[6] for r in rs)}  | "
              + " ".join(f"{r[3]}{'+' if r[4] else '-'}" for r in rs))
    print(f"({time.time()-t:.0f}s)")



