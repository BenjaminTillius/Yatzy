"""
Distributionsmetod för exempelspelet (k=2, n=2, h=2, rho_max=2).
rho_max=2 betyder 2 slag totalt = 1 omslag.
"""
import numpy as np
from itertools import product as iproduct
from math import factorial

N_DICE, N_SIDES, N_CATS = 2, 2, 2
MAX_SCORE = 6

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

T_start  = np.array([multinomial(d) / N_SIDES**N_DICE for d in ALL_DICE], dtype=np.float64)
valid_ki = [np.where(np.all(KEEPS_ARR <= DICE_ARR[di], axis=1))[0] for di in range(ND)]

def score(dice, cat):
    c = list(dice)
    if cat == 0: return c[0] * 1
    if cat == 1: return c[1] * 2
    return 0

score_table = np.array([[score(d,c) for c in range(N_CATS)] for d in ALL_DICE], dtype=np.int32)

N_MASKS = 1 << N_CATS
FULL    = N_MASKS - 1

P_start_all = np.zeros(MAX_SCORE + 1, dtype=np.float64)

for beta in range(MAX_SCORE + 1):
    NT = beta + 1

    round_start_b = np.zeros((N_MASKS, NT), dtype=np.float64)
    round_start_b[FULL, beta] = 1.0

    for mask in range(FULL - 1, -1, -1):
        open_cats = [c for c in range(N_CATS) if not (mask >> c & 1)]
        if not open_cats:
            continue

        V0 = np.zeros((NT, ND), dtype=np.float64)
        for cat in open_cats:
            nm = mask | (1 << cat)
            r  = score_table[:, cat]
            for tau in range(NT):
                new_tau = np.minimum(tau + r, beta).astype(int)
                val     = round_start_b[nm, new_tau]
                np.maximum(V0[tau], val, out=V0[tau])

        # Ett omslag + första kast (rho_max=2)
        for tau in range(NT):
            ev1 = T @ V0[tau]
            V1  = np.array([ev1[valid_ki[di]].max() for di in range(ND)])
            round_start_b[mask, tau] = T_start @ V1

    P_start_all[beta] = round_start_b[0, 0]

print("Sannolikhet att uppnå minst beta poäng från starttillståndet (tau=0):")
for beta in range(MAX_SCORE + 1):
    print(f"  P(>={beta}) = {P_start_all[beta]:.4f}")

print(f"\nFörväntad poäng = {P_start_all[1:].sum():.4f}")
print("(Förväntat: P(>=5) = 0.7734)")