# PSO Benchmark Experiment

Este repositório avalia o desempenho do algoritmo PSO (Particle Swarm Optimization) nas funções **Elliptic** e **Weierstrass** com rotação e deslocamento, utilizando **paralelização de partículas e repetições** para acelerar os testes.

## Funcionalidades
- Benchmark com funções (rotacionadas e deslocadas)
- Otimização com `mealpy`
- Paralelização via `ThreadPoolExecutor` e `multiprocessing`
- Estatísticas descritivas dos resultados
- Curvas de convergência das melhores execuções

## Requisitos
- `mealpy`
- `numpy`
- `pandas`
- `matplotlib`
- `numba`
- `tqdm`

