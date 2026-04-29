"""
Rekordspel: beräknar sannolikheten att uppnå minst BETA poäng i skandinaviskt
Yatzy med optimal strategi.

Belöningsfunktioner (Definition i uppsatsen):
  r_t(s, a) = 0  för alla handlingar utom terminaltillståndet
  r_N(s)    = 1 om tau >= beta, annars 0

Tillstånd: (cats_mask, tau, gamma, dice, rerolls)
  tau:   totalpoäng hittills, cappat vid BETA
  gamma: kvarvarande poäng för bonus (0..63), gamma=0 => bonus säkrad

round_start[mask, tau, gamma] sparas löpande (~2.5 GB för BETA=300).
"""
import numpy as np
from itertools import product as iproduct
from math import factorial

N_DICE, N_SIDES, N_CATS = 5, 6, 15
GAMMA_MAX = 63
NU        = GAMMA_MAX + 1
BONUS     = 50

BETA = 249        # ← ändra detta för att beräkna andra rekordmål
NT   = BETA + 1   # tau i {0, ..., BETA}, allt >= BETA kollapsas till BETA

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

# Förberäkna new_tau[cat][tau, di] = min(tau + score[di,cat], BETA)
tau_arr   = np.arange(NT)[:, None]                    # (NT, 1)
new_tau   = {cat: np.minimum(tau_arr + score_table[:,cat][None,:], BETA).astype(np.int32)
             for cat in range(N_CATS)}                 # (NT, ND)

# Förberäkna new_gamma[cat][gamma, di] = max(0, gamma - score[di,cat])
u_arr     = np.arange(NU)[:, None]
new_gamma = {cat: np.maximum(0, u_arr - score_table[:,cat][None,:]).astype(np.int32)
             for cat in range(6)}                      # (NU, ND)

def best_reroll_batch(V):
    """
    V: (NT, NU, ND) → V_next: (NT, NU, ND)
    Plattar ut (NT, NU) till en dimension för att köra T-matrismultiplikationen
    en gång, loopar sedan bara över di (252 iterationer).
    """
    Vf  = V.reshape(NT * NU, ND)
    EV  = T @ Vf.T                           # (NK, NT*NU)
    V1f = np.empty_like(Vf)
    for di in range(ND):
        V1f[:, di] = EV[valid_ki[di], :].max(axis=0)
    return V1f.reshape(NT, NU, ND)

N_MASKS = 1 << N_CATS
FULL    = N_MASKS - 1

print(f"Allokerar round_start ({N_MASKS}×{NT}×{NU}, "
      f"{N_MASKS*NT*NU*4/1024**3:.2f} GB)...")
round_start = np.zeros((N_MASKS, NT, NU), dtype=np.float32)

# Terminalbelöning: r_N = 1 om tau = BETA (dvs tau >= beta i cappat tillstånd)
round_start[FULL, BETA, :] = 1.0

# ── Bakåtinduktion (~5 timmar för BETA=300) ──────────────────────────────────
print(f"Bakåtinduktion (OBS: tar ~5 timmar för BETA={BETA})...")
for mask in range(FULL - 1, -1, -1):
    open_cats = [c for c in range(N_CATS) if not (mask >> c & 1)]
    if not open_cats:
        continue

    # V0[tau, gamma, di] = max_cat { round_start[new_mask, new_tau, new_gamma] }
    V0 = np.zeros((NT, NU, ND), dtype=np.float32)

    for cat in open_cats:
        nm    = mask | (1 << cat)
        rs_nm = round_start[nm]                        # (NT, NU)

        if cat < 6:
            nt     = new_tau[cat]                      # (NT, ND)
            ng     = new_gamma[cat]                    # (NU, ND)
            nt_b   = np.minimum(nt + BONUS, BETA)      # (NT, ND) — tau med bonus
            nt_3d  = nt[:, None, :]                    # (NT, 1, ND)
            ntb_3d = nt_b[:, None, :]
            ng_3d  = ng[None, :, :]                    # (1, NU, ND)
            new_tau_3d = np.where(ng_3d == 0, ntb_3d, nt_3d)  # (NT, NU, ND)
            rs = rs_nm[new_tau_3d,
                       ng_3d * np.ones((NT, 1, 1), dtype=np.int32)]
        else:
            nt = new_tau[cat]                          # (NT, ND)
            rs = rs_nm[nt[:, None, :],
                       np.arange(NU)[None, :, None]]   # (NT, NU, ND)

        np.maximum(V0, rs, out=V0)

    V1 = best_reroll_batch(V0)
    V2 = best_reroll_batch(V1)

    # round_start[mask, tau, gamma] = E[V2[tau, gamma, first_roll]]
    V2f = V2.reshape(NT * NU, ND)
    round_start[mask] = (T_start @ V2f.T).reshape(NT, NU)

print(f"\nP(>= {BETA} poäng | optimal strategi) = {round_start[0, 0, GAMMA_MAX]:.4f}")

# ── Simulering ────────────────────────────────────────────────────────────────
# get_policy(mask, tau, gamma) beräknar och cachar optimal strategi för ett
# tillstånd. Varje unik (mask, tau, gamma) beräknas bara en gång.

cache = {}

def get_policy(mask, tau, gamma):
    key = (mask, tau, gamma)
    if key in cache:
        return cache[key]
    open_cats = [c for c in range(N_CATS) if not (mask >> c & 1)]
    V0 = np.zeros(ND, dtype=np.float32)
    best_c = np.zeros(ND, dtype=np.int8)
    for cat in open_cats:
        nm = mask | (1 << cat)
        r  = score_table[:, cat]
        if cat < 6:
            ng  = np.maximum(0, gamma - r).astype(int)
            nt  = np.minimum(tau + r, BETA).astype(int)
            nt_b = np.minimum(nt + BONUS, BETA)
            new_t = np.where(ng == 0, nt_b, nt)
            rs = round_start[nm, new_t, ng]
        else:
            nt = np.minimum(tau + r, BETA).astype(int)
            rs = round_start[nm, nt, gamma]
        improved = rs > V0
        V0[improved]     = rs[improved]
        best_c[improved] = cat
    ev1   = T @ V0
    V1    = np.array([ev1[valid_ki[di]].max() for di in range(ND)])
    ev2   = T @ V1
    keep1 = np.array([valid_ki[di][ev1[valid_ki[di]].argmax()] for di in range(ND)], dtype=np.int16)
    keep2 = np.array([valid_ki[di][ev2[valid_ki[di]].argmax()] for di in range(ND)], dtype=np.int16)
    cache[key] = (best_c, keep1, keep2)
    return cache[key]

print("Simulerar 100 000 spel...")
rng     = np.random.default_rng(42)
N_SIM   = 100_000
results = np.zeros(N_SIM, dtype=np.int32)

for sim in range(N_SIM):
    mask, tau, gamma, total = 0, 0, GAMMA_MAX, 0
    for _ in range(N_CATS):
        best_c, keep1, keep2 = get_policy(mask, tau, gamma)

        roll = rng.integers(1, N_SIDES + 1, N_DICE)
        counts = [0] * N_SIDES
        for d in roll: counts[d-1] += 1
        di = DICE_IDX[tuple(counts)]

        # Omslag 2
        ki = int(keep2[di])
        if ALL_KEEPS[ki] != ALL_DICE[di]:
            kept = [v+1 for v in range(N_SIDES) for _ in range(ALL_KEEPS[ki][v])]
            new  = list(kept) + list(rng.integers(1, N_SIDES+1, N_DICE - len(kept)))
            counts = [0] * N_SIDES
            for d in new: counts[d-1] += 1
            di = DICE_IDX[tuple(counts)]

        # Omslag 1
        ki = int(keep1[di])
        if ALL_KEEPS[ki] != ALL_DICE[di]:
            kept = [v+1 for v in range(N_SIDES) for _ in range(ALL_KEEPS[ki][v])]
            new  = list(kept) + list(rng.integers(1, N_SIDES+1, N_DICE - len(kept)))
            counts = [0] * N_SIDES
            for d in new: counts[d-1] += 1
            di = DICE_IDX[tuple(counts)]

        # Välj kategori
        cat    = int(best_c[di])
        r      = int(score_table[di, cat])
        total += r
        if cat < 6:
            gamma = max(0, gamma - r)
        tau   = min(tau + r, BETA)
        mask |= (1 << cat)

    if gamma == 0:
        total += BONUS
    results[sim] = total

print(f"Simulerat medelvärde:       {results.mean():.4f}")
print(f"Median:                     {np.median(results):.4f}")
print(f"Standardavvikelse:          {results.std():.4f}")
print(f"Min / Max:                  {results.min()} / {results.max()}")
over = (results >= BETA).sum()
print(f"Andel spel >= {BETA}p:       {over}/{N_SIM} ({100*over/N_SIM:.2f}%)")