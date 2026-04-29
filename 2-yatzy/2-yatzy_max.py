"""
Exempelspelet från uppsatsen med optimala belöningsfunktioner (maximera förväntad totalpoäng).
  k=2 (tvåsidiga tärningar), n=2 (två tärningar),
  h=2 (kategorier: ettor och tvåor), rho_max=2 (två slag totalt = 1 omslag).

Kategorier:
  0: Ettor — poäng = antal ettor * 1
  1: Tvåor — poäng = antal tvåor * 2
"""
import numpy as np
from itertools import product as iproduct
from math import factorial

N_DICE, N_SIDES, N_CATS = 2, 2, 2
NU = 1  # inget bonussystem

def multinomial(counts):
    n = sum(counts); r = factorial(n)
    for c in counts: r //= factorial(c)
    return r

ALL_DICE  = [c for c in iproduct(range(N_DICE+1), repeat=N_SIDES) if sum(c)==N_DICE]
ALL_KEEPS = [c for c in iproduct(range(N_DICE+1), repeat=N_SIDES) if sum(c)<=N_DICE]
DICE_IDX  = {d:i for i,d in enumerate(ALL_DICE)}
ND, NK    = len(ALL_DICE), len(ALL_KEEPS)
DICE_ARR  = np.array(ALL_DICE,  dtype=np.int8)
KEEPS_ARR = np.array(ALL_KEEPS, dtype=np.int8)

T = np.zeros((NK, ND), dtype=np.float64)
for ki, keep in enumerate(ALL_KEEPS):
    m = N_DICE - sum(keep); tot = N_SIDES**m
    for free in iproduct(range(m+1), repeat=N_SIDES):
        if sum(free) != m: continue
        res = tuple(keep[i]+free[i] for i in range(N_SIDES))
        if res in DICE_IDX:
            T[ki, DICE_IDX[res]] += multinomial(free) / tot

T_start  = np.array([multinomial(d) / N_SIDES**N_DICE for d in ALL_DICE])
valid_ki = [np.where(np.all(KEEPS_ARR <= DICE_ARR[di], axis=1))[0] for di in range(ND)]

def score(dice, cat):
    c = list(dice)
    if cat == 0: return c[0] * 1   # ettor
    if cat == 1: return c[1] * 2   # tvåor
    return 0

score_table = np.array([[score(d,c) for c in range(N_CATS)] for d in ALL_DICE], dtype=np.float64)

def best_reroll_batch(V):
    """V: (NU, ND) -> V_next: (NU, ND)"""
    EV = T @ V.T
    V1 = np.empty((NU, ND))
    for di in range(ND):
        V1[:, di] = EV[valid_ki[di], :].max(axis=0)
    return V1

N_MASKS = 1 << N_CATS
FULL    = N_MASKS - 1

round_start = np.zeros((N_MASKS, NU))

# Bakåtinduktion
print("Bakåtinduktion...")
for mask in range(FULL - 1, -1, -1):
    open_cats = [c for c in range(N_CATS) if not (mask >> c & 1)]
    if not open_cats:
        continue
    V0 = np.full((NU, ND), -1e18)
    for cat in open_cats:
        nm = mask | (1 << cat)
        rs = round_start[nm, :][:, None]
        np.maximum(V0, score_table[:, cat][None,:] + rs, out=V0)
    V1 = best_reroll_batch(V0)
    round_start[mask, :] = T_start @ V1.T

print(f"u*(s_1) = {round_start[0, 0]:.4f}")

# Simulering
cache = {}

def get_policy(mask):
    if mask in cache:
        return cache[mask]
    open_cats = [c for c in range(N_CATS) if not (mask >> c & 1)]
    V0 = np.full(ND, -1e18); best_c = np.zeros(ND, dtype=np.int8)
    for cat in open_cats:
        nm = mask | (1 << cat)
        val = score_table[:, cat] + round_start[nm, 0]
        improved = val > V0
        V0[improved]     = val[improved]
        best_c[improved] = cat
    ev1   = T @ V0
    keep1 = np.array([valid_ki[di][ev1[valid_ki[di]].argmax()] for di in range(ND)], dtype=np.int16)
    cache[mask] = (best_c, keep1)
    return cache[mask]

print("Simulerar 100 000 spel...")
rng     = np.random.default_rng(42)
N_SIM   = 100_000
results = np.zeros(N_SIM, dtype=np.int32)

for sim in range(N_SIM):
    mask, total = 0, 0
    for _ in range(N_CATS):
        best_c, keep1 = get_policy(mask)

        roll = rng.integers(1, N_SIDES + 1, N_DICE)
        counts = [0] * N_SIDES
        for d in roll: counts[d-1] += 1
        di = DICE_IDX[tuple(counts)]

        ki = int(keep1[di])
        if ALL_KEEPS[ki] != ALL_DICE[di]:
            kept = [v+1 for v in range(N_SIDES) for _ in range(ALL_KEEPS[ki][v])]
            new  = list(kept) + list(rng.integers(1, N_SIDES+1, N_DICE - len(kept)))
            counts = [0] * N_SIDES
            for d in new: counts[d-1] += 1
            di = DICE_IDX[tuple(counts)]

        cat    = int(best_c[di])
        r      = int(score_table[di, cat])
        total += r
        mask  |= (1 << cat)

    results[sim] = total

over = (results >= 5).sum()
print(f"Simulerat medelvärde:    {results.mean():.4f}")
print(f"Median:                  {np.median(results):.4f}")
print(f"Standardavvikelse:       {results.std():.4f}")
print(f"Min / Max:               {results.min()} / {results.max()}")
print(f"Andel spel >= 5p:        {over}/{N_SIM} ({100*over/N_SIM:.2f}%)")
