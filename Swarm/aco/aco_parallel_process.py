"""
Experimento: ACOR (MEALPY) em duas funções-teste
–––––––––––––––––––––––––––––––––––––––––––––––––
✓ 30 repetições independentes por (função, configuração)
✓ estatísticas (média, desvio-padrão, mediana) dos melhores fitness
✓ gráfico best-fitness-vs-epoch do melhor run (salvo .png)
✓ domínio [-100, 100] para ambas as funções
✓ parâmetros fixos ρ=0 ·5, α=1, β=2  ➜
      – ρ  → intent_factor  (evaporação de feromônio)
      – α  → zeta           (importância do feromônio)
      – β  → ***não existe*** no ACOR-continuous; não afeta o código, mas
        documentamos a limitação no relatório.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from mealpy import FloatVar, Problem, ACOR
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time

class RotatedElliptic(Problem):
    def __init__(self, n_dims=10, angle=np.pi/4):
        self.angle = angle
        self.R = None
        bounds = FloatVar(lb=(-100.,)*n_dims, ub=(100.,)*n_dims, name="x")
        super().__init__(bounds, minmax="min")
        self.name = "Rotated High Conditioned Elliptic"

    def _rotate(self, x):
        if self.R is None:
            Q, _ = np.linalg.qr(np.random.randn(len(x), len(x)))
            if np.linalg.det(Q) < 0:
                Q[:, 0] = -Q[:, 0]
            self.R = Q
        return self.R @ x

    def obj_func(self, sol):
        n = len(sol)
        x = self._rotate(np.asarray(sol))
        return np.sum((10**6) ** (np.arange(n)/(n-1)) * x**2)

class ShiftedRotWeierstrass(Problem):
    def __init__(self, n_dims=10, angle=np.pi/6):
        self.angle = angle
        self.shift = np.random.uniform(-0.4, 0.4, n_dims)
        self.R = None
        bounds = FloatVar(lb=(-100.,)*n_dims, ub=(100.,)*n_dims, name="x")
        super().__init__(bounds, minmax="min")
        self.name = "Shifted & Rotated Weierstrass"

    def _rotate(self, x):
        if self.R is None:
            Q, _ = np.linalg.qr(np.random.randn(len(x), len(x)))
            if np.linalg.det(Q) < 0:
                Q[:, 0] = -Q[:, 0]
            self.R = Q
        return self.R @ x

    def obj_func(self, sol):
        a, b, kmax = 0.5, 3, 20
        x = self._rotate(np.asarray(sol) - self.shift)
        term1 = np.sum([
            np.sum(a**k * np.cos(2*np.pi*b**k*(x+0.5)))
            for k in range(kmax+1)
        ])
        term0 = (len(x) *
                 np.sum([a**k * np.cos(2*np.pi*b**k*0.5)
                         for k in range(kmax+1)]))
        return term1 - term0

CONFIGS = [
    {"id": 1, "pop": 20,  "epoch": 500},
    {"id": 2, "pop": 50,  "epoch": 1000},
    {"id": 3, "pop": 100, "epoch": 2000},
]
N_RUNS = 1
COMMON = dict(
    sample_count=25,
    intent_factor=0.5,   # ρ
    zeta=1.0            # α
    # β não existe explicitamente no ACOR-continuous
)

PROBLEMS = [RotatedElliptic(), ShiftedRotWeierstrass()]
results = [] 

plots_dir = Path("acor_plots")
plots_dir.mkdir(exist_ok=True)

def run_once(problem, cfg, seed=None):
    if seed is not None:
        np.random.seed(seed)
    model = ACOR.OriginalACOR(epoch=cfg["epoch"],
                              pop_size=cfg["pop"],
                              **COMMON)
    gbest = model.solve(problem)
    return gbest.target.fitness, model.history.list_global_best_fit

def run_job(prob_cls, cfg, seed):
    """Cria uma nova instância do problema dentro do processo."""
    prob = prob_cls()                  
    return run_once(prob, cfg, seed)

if __name__ == "__main__":            
    all_jobs = []
    future_to_key = {}  
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as pool:
        for prob in PROBLEMS:
            for cfg in CONFIGS:
                for seed in range(N_RUNS):
                    fut = pool.submit(run_job, prob.__class__, cfg, seed)
                    key = (prob.name, cfg["id"])
                    future_to_key[fut] = key
                    all_jobs.append(fut)

        results_map = {}  # chave: (func, cfg_id)  →  listas
        for fut in as_completed(all_jobs):
            prob_name, cfg_id = future_to_key[fut]
            fitness, hist = fut.result()
            key = (prob_name, cfg_id)
            results_map.setdefault(key, []).append((fitness, hist))

        for prob in PROBLEMS:
            for cfg in CONFIGS:
                key = (prob.name, cfg["id"])
                data = results_map.get(key, [])
                if not data:
                    continue
                cfg_start = time.time()
                best_vals, histories = zip(*data)
                best_vals = np.array(best_vals)
                mean, std, med = best_vals.mean(), best_vals.std(ddof=1), np.median(best_vals)
                cfg_time = time.time() - cfg_start  
                results.append((prob.name, cfg["id"], cfg["pop"], cfg["epoch"],
                                mean, std, med, cfg_time))
                print(f"\n### {prob.name} - Config {cfg['id']}  ➜  "
                      f"mean={mean:.3e}, std={std:.3e}, median={med:.3e}, tempo={cfg_time:.2f}s")
                best_run = int(np.argmin(best_vals))
                plt.figure()
                plt.plot(histories[best_run])
                plt.title(f"{prob.name}\nConfig {cfg['id']}  (best of 30 runs)")
                plt.xlabel("Epoch"); plt.ylabel("Best fitness")
                plt.grid()
                fname = plots_dir / f"{prob.name.replace(' ','_')}_cfg{cfg['id']}_parallel_process.png"
                plt.savefig(fname, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"   ↳ curva salva em: {fname}")

        df = pd.DataFrame(results, columns=[
            "Function", "Cfg", "PopSize", "Epochs",
            "MeanBest", "StdBest", "MedianBest", "Seconds"
        ])
        df.to_csv("acor_summary_parallel_process.csv", index=False)
        print("\nResumo salvo em acor_summary_parallel_process.csv")
        print(df)
