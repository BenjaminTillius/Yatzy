"""
Rekordspel: beräknar sannolikheten att uppnå minst BETA poäng i skandinaviskt
Yatzy med optimal strategi.

Fix: bonus läggs bara till när gamma övergår från >0 till 0,
inte varje gång en övre kategori stängs när gamma redan är 0.
"""
import numpy as np
from itertools import product as iproduct
from math import factorial

N_DICE, N_SIDES, N_CATS = 5, 6, 15
GAMMA_MAX = 63
NU        = GAMMA_MAX + 1
BONUS     = 50

BETA = 374
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

T = np.zeros((NK, ND), dtype=np.float32)
for ki, keep in enumerate(ALL_KEEPS):
    m = N_DICE - sum(keep); tot = N_SIDES**m
    for free in iproduct(range(m+1), repeat=N_SIDES):
        if sum(free) != m: continue
        res = tuple(keep[i]+free[i] for i in range(N_SIDES))
        if res in DICE_IDX:
            T[ki, DICE_IDX[res]] += multinomial(free) / tot

T_start  = np.array([multinomial(d) / N_SIDES**N_DICE for d in ALL_DICE], dtype=np.float32)
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

score_table = np.array([[score(d,c) for c in range(N_CATS)] for d in ALL_DICE], dtype=np.float32)

tau_arr   = np.arange(NT)[:, None]
new_tau   = {cat: np.minimum(tau_arr + score_table[:,cat][None,:], BETA).astype(np.int32)
             for cat in range(N_CATS)}

u_arr     = np.arange(NU)[:, None]
new_gamma = {cat: np.maximum(0, u_arr - score_table[:,cat][None,:]).astype(np.int32)
             for cat in range(6)}

def best_reroll_batch(V):
    Vf  = V.reshape(NT * NU, ND)
    EV  = T @ Vf.T
    V1f = np.empty_like(Vf)
    for di in range(ND):
        V1f[:, di] = EV[valid_ki[di], :].max(axis=0)
    return V1f.reshape(NT, NU, ND)

N_MASKS = 1 << N_CATS
FULL    = N_MASKS - 1

print(f"Allokerar round_start ({N_MASKS}×{NT}×{NU}, "
      f"{N_MASKS*NT*NU*4/1024**3:.2f} GB)...")
round_start = np.zeros((N_MASKS, NT, NU), dtype=np.float32)
round_start[FULL, BETA, :] = 1.0

print(f"Bakåtinduktion för BETA={BETA}...")
for mask in range(FULL - 1, -1, -1):
    open_cats = [c for c in range(N_CATS) if not (mask >> c & 1)]
    if not open_cats:
        continue

    V0 = np.zeros((NT, NU, ND), dtype=np.float32)

    for cat in open_cats:
        nm    = mask | (1 << cat)
        rs_nm = round_start[nm]

        if cat < 6:
            nt    = new_tau[cat]           # (NT, ND)
            ng    = new_gamma[cat]         # (NU, ND)
            nt_b  = np.minimum(nt + BONUS, BETA)  # (NT, ND)

            nt_3d  = nt[:, None, :]        # (NT, 1, ND)
            ntb_3d = nt_b[:, None, :]      # (NT, 1, ND)
            ng_3d  = ng[None, :, :]        # (1, NU, ND)

            # Förberäkna gamma_before: de ursprungliga gamma-värdena (0..63)
            gamma_before = np.arange(NU)[None, :, None]  # (1, NU, 1)

            # Bonus läggs bara till när gamma övergår från >0 till 0
            bonus_trigger = (gamma_before > 0) & (ng_3d == 0)  # (1, NU, ND) -> (NT, NU, ND) via broadcast

            new_tau_3d = np.where(bonus_trigger, ntb_3d, nt_3d)  # (NT, NU, ND)

            rs = rs_nm[new_tau_3d,
                       ng_3d * np.ones((NT, 1, 1), dtype=np.int32)]
        else:
            nt = new_tau[cat]
            rs = rs_nm[nt[:, None, :],
                       np.arange(NU)[None, :, None]]

        np.maximum(V0, rs, out=V0)

    V1 = best_reroll_batch(V0)
    V2 = best_reroll_batch(V1)

    V2f = V2.reshape(NT * NU, ND)
    round_start[mask] = (T_start @ V2f.T).reshape(NT, NU)

print(f"\nP(>= {BETA} poäng | optimal rekordspelsstrategi) = {round_start[0, 0, GAMMA_MAX]:.4f}")