"""
Beräknar väntevärdet u*(s_1) för skandinaviskt Yatzy med bonus.

Tillstånd: (cats_mask, gamma, dice, rerolls)
  gamma = min(återstående poäng för bonus, 63), gamma=0 => bonus säkrad
  cats_mask: 15-bitars heltal, bit c = 1 om kategori c är stängd
  dice: index i ALL_DICE (252 möjliga utfall)
  rerolls: 0, 1 eller 2

round_start[mask, gamma] = E[u*(nästa runda)] = T_start @ V[mask, gamma, :, 2]
sparas löpande (32768 × 64, 16 MB).
"""
import numpy as np
from itertools import product as iproduct
from math import factorial

N_DICE, N_SIDES, N_CATS = 5, 6, 15
GAMMA_MAX = 63
NU = GAMMA_MAX + 1  # gamma in {0, ..., 63}
BONUS = 50

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

# Övergångsmatris T[ki, di] = P(result=di | behåll keep ki, kasta om resten)
T = np.zeros((NK, ND))
for ki, keep in enumerate(ALL_KEEPS):
    m = N_DICE - sum(keep); tot = N_SIDES**m
    for free in iproduct(range(m+1), repeat=N_SIDES):
        if sum(free) != m: continue
        res = tuple(keep[i]+free[i] for i in range(N_SIDES))
        if res in DICE_IDX:
            T[ki, DICE_IDX[res]] += multinomial(free) / tot

T_start = np.array([multinomial(d) / N_SIDES**N_DICE for d in ALL_DICE])

# valid_ki[di] = index till giltiga keeps för tärningsutfall di
valid = np.all(KEEPS_ARR[None,:,:] <= DICE_ARR[:,None,:], axis=2)
valid_ki = [np.where(valid[di])[0] for di in range(ND)]

# Poängtabell
def score(dice, cat):
    c = list(dice)
    if cat <= 5: return c[cat] * (cat + 1)
    if cat == 6:
        for v in range(5,-1,-1):
            if c[v] >= 2: return 2*(v+1)
        return 0
    if cat == 7:
        pairs = [v+1 for v in range(6) if c[v] >= 2]
        return 2*sum(sorted(pairs)[-2:]) if len(pairs) >= 2 else 0
    if cat == 8:
        for v in range(5,-1,-1):
            if c[v] >= 3: return 3*(v+1)
        return 0
    if cat == 9:
        for v in range(5,-1,-1):
            if c[v] >= 4: return 4*(v+1)
        return 0
    if cat == 10: return 15 if all(c[v] >= 1 for v in range(5)) else 0
    if cat == 11: return 20 if all(c[v] >= 1 for v in range(1,6)) else 0
    if cat == 12:
        three = next((v+1 for v in range(5,-1,-1) if c[v] >= 3), None)
        two   = next((v+1 for v in range(5,-1,-1) if c[v] >= 2 and (v+1) != three), None)
        return 3*three + 2*two if three and two else 0
    if cat == 13: return sum((v+1)*c[v] for v in range(6))
    if cat == 14: return 50 if any(x == 5 for x in c) else 0
    return 0

score_table = np.array([[score(d,c) for c in range(N_CATS)] for d in ALL_DICE], dtype=np.float64)

# gamma' = max(0, gamma - r) för övre kategorier (cat 0-5)
u_arr = np.arange(NU)[:,None]
new_gamma = {cat: np.maximum(0, u_arr - score_table[:,cat][None,:]).astype(np.int32)
             for cat in range(6)}

def best_reroll_batch(V0):
    """V0: (NU, ND) → V1: (NU, ND)"""
    EV = T @ V0.T  # (NK, NU)
    V1 = np.empty((NU, ND))
    for di in range(ND):
        V1[:, di] = EV[valid_ki[di], :].max(axis=0)
    return V1

# round_start[mask, gamma] = T_start @ V[mask, gamma, :, 2]
N_MASKS = 1 << N_CATS
FULL    = N_MASKS - 1
round_start = np.zeros((N_MASKS, NU))

# Terminalbelöning: r_N(s) = 50 om gamma=0, annars 0 (Definition 3)
round_start[FULL, 0] = BONUS

# Bakåtinduktion
for mask in range(FULL - 1, -1, -1):
    open_cats = [c for c in range(N_CATS) if not (mask >> c & 1)]
    if not open_cats:
        continue

    # Kategorival: V0[gamma, di] = max_cat { r_t(s,a) + round_start[new_mask, gamma'] }
    V0 = np.full((NU, ND), 0.0)
    for cat in open_cats:
        nm = mask | (1 << cat)
        r  = score_table[:, cat]                         # (ND,)
        if cat < 6:
            ng = new_gamma[cat]                          # (NU, ND)
            rs = round_start[nm, ng]                     # (NU, ND)
        else:
            rs = round_start[nm, :][:,None]              # (NU, 1) → broadcast
        np.maximum(V0, r[None,:] + rs, out=V0)

    V1 = best_reroll_batch(V0)
    V2 = best_reroll_batch(V1)
    round_start[mask, :] = T_start @ V2.T

# u*(s_1): mask=0, gamma=63, kasta alla 5 tärningar, sedan 2 omslag
print(f"u*(s_1) = {round_start[0, GAMMA_MAX]:.4f}")