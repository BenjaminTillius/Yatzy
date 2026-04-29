"""
Beräknar E[poäng | strategi Rekord] för exempelspelet.
  k=2, n=2, h=2, rho_max=2 (1 omslag), beta=5.

Steg 1: Beräkna strategi Rekord (maximera P(tau>=5)) och spara alla handlingar.
Steg 2: Policy evaluation — följ strategi Rekord men utvärdера med
        optimala belöningsfunktionerna (r_t = poäng i kategori).
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
    if cat == 0: return c[0] * 1
    if cat == 1: return c[1] * 2
    return 0

score_table = np.array([[score(d,c) for c in range(N_CATS)] for d in ALL_DICE], dtype=np.float64)

N_MASKS = 1 << N_CATS
FULL    = N_MASKS - 1

# ── Steg 1: Beräkna och spara strategi Rekord ────────────────────────────────
round_start_rek = np.zeros((N_MASKS, NT))
best_cat_rek    = np.zeros((N_MASKS, NT, ND), dtype=np.int8)
best_keep_rek   = np.zeros((N_MASKS, NT, ND), dtype=np.int16)
round_start_rek[FULL, BETA] = 1.0

for mask in range(FULL - 1, -1, -1):
    open_cats = [c for c in range(N_CATS) if not (mask >> c & 1)]
    if not open_cats:
        continue
    V0 = np.zeros((NT, ND)); best_c = np.zeros((NT, ND), dtype=np.int8)
    for cat in open_cats:
        nm = mask | (1 << cat); r = score_table[:, cat]
        for tau in range(NT):
            new_tau  = np.minimum(tau + r, BETA).astype(int)
            val      = round_start_rek[nm, new_tau]
            improved = val > V0[tau]
            V0[tau, improved] = val[improved]; best_c[tau, improved] = cat
    best_cat_rek[mask] = best_c
    for tau in range(NT):
        ev1 = T @ V0[tau]
        best_keep_rek[mask, tau] = np.array(
            [valid_ki[di][ev1[valid_ki[di]].argmax()] for di in range(ND)], dtype=np.int16)
        round_start_rek[mask, tau] = T_start @ np.array(
            [ev1[valid_ki[di]].max() for di in range(ND)])

print(f"u*(s_1) Rekord = {round_start_rek[0, 0]:.4f}")

# ── Steg 2: Policy evaluation med optimala belöningsfunktioner ───────────────
round_start_eval = np.zeros((N_MASKS, NT))

for mask in range(FULL - 1, -1, -1):
    open_cats = [c for c in range(N_CATS) if not (mask >> c & 1)]
    if not open_cats:
        continue

    # Kategorival: följ best_cat_rek, belöning = poäng + framtidsvärde
    V0 = np.zeros((NT, ND))
    for tau in range(NT):
        for di in range(ND):
            cat     = int(best_cat_rek[mask, tau, di])
            nm      = mask | (1 << cat)
            r       = int(score_table[di, cat])
            new_tau = min(tau + r, BETA)
            V0[tau, di] = r + round_start_eval[nm, new_tau]

    # Omslagsval: följ best_keep_rek
    V1 = np.zeros((NT, ND))
    for tau in range(NT):
        for di in range(ND):
            ki = int(best_keep_rek[mask, tau, di])
            V1[tau, di] = T[ki, :] @ V0[tau, :]

    for tau in range(NT):
        round_start_eval[mask, tau] = T_start @ V1[tau]

print(f"E[poäng | strategi Rekord] = {round_start_eval[0, 0]:.4f}")