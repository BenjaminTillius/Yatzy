"""
Rekordspel: beräknar sannolikheten att uppnå minst k poäng för ALLA k simultant,
enligt Pawlewicz distributionsmetod. Fullt vektoriserad implementation.

För varje tillstånd (mask, gamma) lagras en fördelning P där:
  P[k] = sannolikheten att uppnå minst k poäng från detta tillstånd med optimal strategi.

Minnesanvändning: N_MASKS * NU * N_SCORES * 4 bytes
  = 32768 * 64 * 401 * 4 ≈ 3.4 GB
"""
import numpy as np
from itertools import product as iproduct
from math import factorial

N_DICE, N_SIDES, N_CATS = 5, 6, 15
GAMMA_MAX = 63
NU        = GAMMA_MAX + 1
BONUS     = 50
MAX_SCORE = 400
N_SCORES  = MAX_SCORE + 1   # index 0..MAX_SCORE

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

# ── Förberäkna keep_to_dice: för varje keep ki, vilka di är möjliga utfall? ──
# Vi bygger en (NK, ND) boolean-matris där entry [ki, di] = True om di är ett
# möjligt utfall från keep ki. Detta är transposen av valid_ki-strukturen.
# Används för att vektorisera maximum-operationen.
keep_to_dice = np.zeros((NK, ND), dtype=bool)
for di in range(ND):
    keep_to_dice[valid_ki[di], di] = True

# För varje di: vilka ki är giltiga? (samma som valid_ki men som matris)
# Vi förberäknar också en mappning di -> keep_index för maximum-operationen.
# Strategi: för varje di beräknar vi maximum över valid_ki[di] i EV.
# Vi använder np.maximum.reduceat med sorterade index.

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

score_table = np.array([[score(d,c) for c in range(N_CATS)] for d in ALL_DICE], dtype=np.int32)

# Förberäkna new_gamma[cat][gamma, di] = max(0, gamma - score[di,cat])
u_arr     = np.arange(NU)[:, None]
new_gamma = {cat: np.maximum(0, u_arr - score_table[:,cat][None,:]).astype(np.int32)
             for cat in range(6)}

N_MASKS = 1 << N_CATS
FULL    = N_MASKS - 1

# ── Allokera round_start[mask, gamma, score_idx] ─────────────────────────────
mem_gb = N_MASKS * NU * N_SCORES * 4 / 1024**3
print(f"Allokerar round_start ({N_MASKS}x{NU}x{N_SCORES}, {mem_gb:.2f} GB)...")
round_start = np.zeros((N_MASKS, NU, N_SCORES), dtype=np.float32)

# ── Terminalfördelning ────────────────────────────────────────────────────────
# gamma=0: bonus säkrad, P[k]=1 för k in {0,...,BONUS}
round_start[FULL, 0, :BONUS + 1] = 1.0
# gamma>0: ingen bonus, P[k]=1 för k=0
round_start[FULL, 1:, 0] = 1.0

# ── Shift-operation ───────────────────────────────────────────────────────────
def shift_dist(P, r):
    """
    Shift P r steg åt höger längs sista axeln.
    (P shift r)[..., k] = P[..., k-r] för k>=r, 0 annars.
    P: godtycklig form med N_SCORES som sista dimension.
    """
    if r == 0:
        return P
    result = np.zeros_like(P)
    result[..., r:] = P[..., :N_SCORES - r]
    return result

# ── Vektoriserad omslagsberäkning ─────────────────────────────────────────────
def best_reroll_batch_dist(V):
    """
    V: (NU, ND, N_SCORES)
    Returnerar V1: (NU, ND, N_SCORES)

    Steg 1 — Viktat medelvärde (slumphändelse, kast av tärningar):
      EV[g, ki, s] = sum_di T[ki, di] * V[g, di, s]
      Beräknas som einsum('kd,gds->gks', T, V)

    Steg 2 — Maximum (spelarens val av vilka tärningar att behålla):
      V1[g, di, s] = max_{ki in valid_ki[di]} EV[g, ki, s]
      Vektoriseras med en di->ki mappningsmatris.
    """
    # Steg 1: EV[g, ki, s] = sum_d T[ki,d] * V[g,d,s]
    # einsum är exakt men kan vara långsamt för stora tensorer.
    # Alternativt: omforma och använd matmultiplikation.
    # V: (NU, ND, N_SCORES) -> (NU*N_SCORES, ND)^T * T^T
    # EV: (NK, NU*N_SCORES) -> (NU, NK, N_SCORES)
    V_flat = V.reshape(NU, ND, N_SCORES)
    # T: (NK, ND), V[g]: (ND, N_SCORES) -> EV_g: (NK, N_SCORES)
    # Batcha över g: V_perm: (ND, NU*N_SCORES) -> T @ V_perm: (NK, NU*N_SCORES)
    V_perm = V_flat.transpose(1, 0, 2).reshape(ND, NU * N_SCORES)  # (ND, NU*N_SCORES)
    EV_flat = T @ V_perm                                             # (NK, NU*N_SCORES)
    EV = EV_flat.reshape(NK, NU, N_SCORES).transpose(1, 0, 2)       # (NU, NK, N_SCORES)

    # Steg 2: för varje di, ta max över valid_ki[di] i EV
    # Vi loopar bara över ND=252 di-värden (inte NU*ND som tidigare)
    V1 = np.empty((NU, ND, N_SCORES), dtype=np.float32)
    for di in range(ND):
        V1[:, di, :] = EV[:, valid_ki[di], :].max(axis=1)  # (NU, N_SCORES)
    return V1

# ── Bakåtinduktion ────────────────────────────────────────────────────────────
print("Bakåtinduktion...")
for mask in range(FULL - 1, -1, -1):
    open_cats = [c for c in range(N_CATS) if not (mask >> c & 1)]
    if not open_cats:
        continue

    # V0[gamma, di, k] = max_cat fördelning efter optimalt kategorival
    V0 = np.zeros((NU, ND, N_SCORES), dtype=np.float32)

    for cat in open_cats:
        nm = mask | (1 << cat)

        if cat < 6:
            # Övre sektion: gamma uppdateras
            # new_gamma[cat]: (NU, ND) — nytt gamma för varje (gamma, di)
            ng = new_gamma[cat]                        # (NU, ND)
            # Hämta nästa fördelning för alla (gamma, di) simultant
            P_next = round_start[nm, ng, :]            # (NU, ND, N_SCORES)
            # Shift med poäng per di — score_table[:, cat] är (ND,)
            # Olika di ger olika shift, måste loopa över unika poängvärden
            unique_scores = np.unique(score_table[:, cat])
            P_shifted = np.zeros_like(P_next)
            for ri in unique_scores:
                di_mask = (score_table[:, cat] == ri)  # (ND,) boolean
                if di_mask.any():
                    P_shifted[:, di_mask, :] = shift_dist(P_next[:, di_mask, :], int(ri))
            np.maximum(V0, P_shifted, out=V0)
        else:
            # Undre sektion: gamma påverkas ej
            # round_start[nm]: (NU, N_SCORES) — samma för alla di med samma poäng
            unique_scores = np.unique(score_table[:, cat])
            for ri in unique_scores:
                di_mask = (score_table[:, cat] == ri)  # (ND,) boolean
                if di_mask.any():
                    P_next    = round_start[nm, :, :]              # (NU, N_SCORES)
                    P_shifted = shift_dist(P_next, int(ri))        # (NU, N_SCORES)
                    # Broadcast till (NU, ND_subset, N_SCORES) och ta maximum
                    np.maximum(
                        V0[:, di_mask, :],
                        P_shifted[:, None, :],
                        out=V0[:, di_mask, :]
                    )

    # Två omslag bakåt
    V1 = best_reroll_batch_dist(V0)
    V2 = best_reroll_batch_dist(V1)

    # Förväntad fördelning över första kastet
    # round_start[mask, g, :] = sum_di T_start[di] * V2[g, di, :]
    # T_start: (ND,), V2: (NU, ND, N_SCORES)
    # -> (NU, N_SCORES) via einsum eller matmultiplikation
    round_start[mask] = np.einsum('d,gds->gs', T_start, V2)

    if mask % 1000 == 0:
        print(f"  mask {mask}/{FULL}...")

# ── Resultat ──────────────────────────────────────────────────────────────────
print("\nSannolikhet att uppnå minst k poäng från starttillståndet:")
P_start = round_start[0, GAMMA_MAX, :]
print(f"{'k':<6} {'P(>=k)':<10}")
print("-" * 16)
for k in range(0, MAX_SCORE + 1, 10):
    print(f"{k:<6} {P_start[k]:.4f}")

print()
for beta in [200, 225, 248, 249, 250, 275, 300]:
    if beta <= MAX_SCORE:
        print(f"P(>= {beta}) = {P_start[beta]:.4f}")