
"""
Beräknar väntevärdet u*(s_1) för skandinaviskt Yatzy med bonus
och simulerar 100 000 spel med den optimala strategin.
"""
import numpy as np
from itertools import product as iproduct
from math import factorial

N_DICE, N_SIDES, N_CATS = 5, 6, 15
GAMMA_MAX = 63
NU    = GAMMA_MAX + 1
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

score_table = np.array([[score(d,c) for c in range(N_CATS)] for d in ALL_DICE])

u_arr     = np.arange(NU)[:, None]
new_gamma = {cat: np.maximum(0, u_arr - score_table[:,cat][None,:]).astype(np.int32)
             for cat in range(6)}

def best_reroll_batch(V):
    """V: (NU, ND) → V_next: (NU, ND)"""
    EV = T @ V.T
    V1 = np.empty((NU, ND))
    for di in range(ND):
        V1[:, di] = EV[valid_ki[di], :].max(axis=0)
    return V1

N_MASKS = 1 << N_CATS
FULL    = N_MASKS - 1

round_start = np.zeros((N_MASKS, NU))
round_start[FULL, 0] = BONUS

# ── Bakåtinduktion ────────────────────────────────────────────────────────────
print("Bakåtinduktion (~3 min)...")
for mask in range(FULL - 1, -1, -1):
    open_cats = [c for c in range(N_CATS) if not (mask >> c & 1)]
    if not open_cats:
        continue
    V0 = np.full((NU, ND), -1e18)
    for cat in open_cats:
        nm  = mask | (1 << cat)
        r   = score_table[:, cat]
        rs  = round_start[nm, new_gamma[cat]] if cat < 6 else round_start[nm, :][:, None]
        np.maximum(V0, r[None,:] + rs, out=V0)
    V1 = best_reroll_batch(V0)
    V2 = best_reroll_batch(V1)
    round_start[mask, :] = T_start @ V2.T

print(f"u*(s_1) = {round_start[0, GAMMA_MAX]:.4f}")

# ── Förberäkna policy för alla maskar ────────────────────────────────────────
print("Förberäknar policy (~2 min)...")
policy_cat   = np.zeros((N_MASKS, NU, ND), dtype=np.int8)
policy_keep1 = np.zeros((N_MASKS, NU, ND), dtype=np.int16)
policy_keep2 = np.zeros((N_MASKS, NU, ND), dtype=np.int16)

for mask in range(FULL):
    open_cats = [c for c in range(N_CATS) if not (mask >> c & 1)]
    if not open_cats:
        continue
    # Kategorival: batch över alla gamma parallellt
    V0 = np.full((NU, ND), -1e18); best_c = np.zeros((NU, ND), dtype=np.int8)
    for cat in open_cats:
        nm  = mask | (1 << cat)
        r   = score_table[:, cat]
        rs  = round_start[nm, new_gamma[cat]] if cat < 6 else round_start[nm, :][:, None]
        val = r[None,:] + rs; imp = val > V0
        V0 = np.where(imp, val, V0); best_c = np.where(imp, np.int8(cat), best_c)
    policy_cat[mask] = best_c
    # Omslagsval: EV (NK, NU), argmax per di över giltiga keeps
    EV1 = T @ V0.T
    V1  = np.empty((NU, ND))
    for di in range(ND): V1[:, di] = EV1[valid_ki[di], :].max(axis=0)
    EV2 = T @ V1.T
    policy_keep1[mask] = np.array([valid_ki[di][EV1[valid_ki[di], :].argmax(axis=0)] for di in range(ND)], dtype=np.int16).T
    policy_keep2[mask] = np.array([valid_ki[di][EV2[valid_ki[di], :].argmax(axis=0)] for di in range(ND)], dtype=np.int16).T

print("Simulerar 100 000 spel...")
rng     = np.random.default_rng(42)
N_SIM   = 100_000
results = np.zeros(N_SIM, dtype=np.int32)

for sim in range(N_SIM):
    mask, gamma, total = 0, GAMMA_MAX, 0
    for _ in range(N_CATS):
        roll = rng.integers(1, N_SIDES + 1, N_DICE)
        counts = [0] * N_SIDES
        for d in roll: counts[d-1] += 1
        di = DICE_IDX[tuple(counts)]

        for keep_arr in [policy_keep2, policy_keep1]:
            ki = int(keep_arr[mask, gamma, di])
            if ALL_KEEPS[ki] != ALL_DICE[di]:
                kept = [v+1 for v in range(N_SIDES) for _ in range(ALL_KEEPS[ki][v])]
                new  = list(kept) + list(rng.integers(1, N_SIDES+1, N_DICE - len(kept)))
                counts = [0] * N_SIDES
                for d in new: counts[d-1] += 1
                di = DICE_IDX[tuple(counts)]

        cat    = int(policy_cat[mask, gamma, di])
        r      = int(score_table[di, cat])
        total += r
        if cat < 6:
            gamma = max(0, gamma - r)
        mask |= (1 << cat)

    if gamma == 0:
        total += BONUS
    results[sim] = total

print(f"Simulerat medelvärde: {results.mean():.4f}")
print(f"Median:               {np.median(results):.4f}")
print(f"Standardavvikelse:    {results.std():.4f}")
print(f"Min / Max:            {results.min()} / {results.max()}")
over_300 = (results >= 300).sum()
print(f"Andel spel >= 300p:   {over_300}/{N_SIM} ({100*over_300/N_SIM:.2f}%)")