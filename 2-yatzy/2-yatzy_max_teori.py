"""
Beräknar P(>= 5 poäng | strategi Optimal) för exempelspelet.
  k=2, n=2, h=2, rho_max=2 (1 omslag), beta=5.

Steg 1: Beräkna strategi Optimal (maximera förväntad poäng) och spara alla handlingar.
Steg 2: Policy evaluation — följ strategi Optimal men utvärdера med
        rekordbelöningsfunktionerna (r_N=1 om tau>=5, annars 0).
"""
import numpy as np
from itertools import product as iproduct
from math import factorial

N_DICE, N_SIDES, N_CATS = 2, 2, 2
BETA = 5
NT   = BETA + 1

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

T = np.zeros((NK, ND))
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

N_MASKS = 1 << N_CATS
FULL    = N_MASKS - 1

# ── Steg 1: Beräkna och spara strategi Optimal ───────────────────────────────
round_start_opt  = np.zeros(N_MASKS)
best_cat_opt     = np.zeros((N_MASKS, ND), dtype=np.int8)
best_keep_opt    = np.zeros((N_MASKS, ND), dtype=np.int16)

for mask in range(FULL - 1, -1, -1):
    open_cats = [c for c in range(N_CATS) if not (mask >> c & 1)]
    if not open_cats:
        continue
    V0 = np.full(ND, -1e18); best_c = np.zeros(ND, dtype=np.int8)
    for cat in open_cats:
        nm  = mask | (1 << cat)
        val = score_table[:, cat] + round_start_opt[nm]
        improved = val > V0; V0[improved] = val[improved]; best_c[improved] = cat
    best_cat_opt[mask] = best_c
    ev1 = T @ V0
    best_keep_opt[mask] = np.array([valid_ki[di][ev1[valid_ki[di]].argmax()] for di in range(ND)], dtype=np.int16)
    V1 = np.array([ev1[valid_ki[di]].max() for di in range(ND)])
    round_start_opt[mask] = T_start @ V1

# ── Steg 2: Policy evaluation med rekordbelöningsfunktioner ──────────────────
round_start_eval = np.zeros((N_MASKS, NT))
round_start_eval[FULL, BETA] = 1.0   # r_N = 1 om tau >= beta

for mask in range(FULL - 1, -1, -1):
    open_cats = [c for c in range(N_CATS) if not (mask >> c & 1)]
    if not open_cats:
        continue

    # Kategorival: följ best_cat_opt, utvärdера med rekordbelöning
    V0 = np.zeros((NT, ND))
    for di in range(ND):
        cat     = int(best_cat_opt[mask, di])
        nm      = mask | (1 << cat)
        r       = int(score_table[di, cat])
        for tau in range(NT):
            new_tau        = min(tau + r, BETA)
            V0[tau, di]    = round_start_eval[nm, new_tau]

    # Omslagsval: följ best_keep_opt
    V1 = np.zeros((NT, ND))
    for di in range(ND):
        ki = int(best_keep_opt[mask, di])
        for tau in range(NT):
            V1[tau, di] = T[ki, :] @ V0[tau, :]

    for tau in range(NT):
        round_start_eval[mask, tau] = T_start @ V1[tau]

print(f"P(>= {BETA} | strategi Optimal) = {round_start_eval[0, 0]:.4f}")
