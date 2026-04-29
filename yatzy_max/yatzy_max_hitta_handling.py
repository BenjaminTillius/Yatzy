"""
Beräknar optimal handling givet ett tillstånd i skandinaviskt Yatzy med bonus.

Indata: tillstånd s = (delta, kappa, rho, gamma)
  - delta: tuple av längd 6, antal tärningar av varje valör (1-6)
  - kappa: tuple av längd 15, 0=öppen 1=stängd för varje kategori
  - rho:   antal återstående omslag (0, 1 eller 2)
  - gamma: poäng kvar till bonus (0-63)

Notera: tau (totalpoäng) behövs inte för pi^Optimal.

Utdata:
  - Om rho=1 eller rho=2: vilka tärningar som ska behållas
  - Om rho=0: vilken kategori som ska väljas
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

# ── Förberäkna policy ─────────────────────────────────────────────────────────
print("Förberäknar policy (~2 min)...")
policy_cat   = np.zeros((N_MASKS, NU, ND), dtype=np.int8)
policy_keep1 = np.zeros((N_MASKS, NU, ND), dtype=np.int16)
policy_keep2 = np.zeros((N_MASKS, NU, ND), dtype=np.int16)

for mask in range(FULL):
    open_cats = [c for c in range(N_CATS) if not (mask >> c & 1)]
    if not open_cats:
        continue
    V0 = np.full((NU, ND), -1e18); best_c = np.zeros((NU, ND), dtype=np.int8)
    for cat in open_cats:
        nm  = mask | (1 << cat)
        r   = score_table[:, cat]
        rs  = round_start[nm, new_gamma[cat]] if cat < 6 else round_start[nm, :][:, None]
        val = r[None,:] + rs; imp = val > V0
        V0 = np.where(imp, val, V0); best_c = np.where(imp, np.int8(cat), best_c)
    policy_cat[mask] = best_c
    EV1 = T @ V0.T
    V1  = np.empty((NU, ND))
    for di in range(ND): V1[:, di] = EV1[valid_ki[di], :].max(axis=0)
    EV2 = T @ V1.T
    policy_keep1[mask] = np.array([valid_ki[di][EV1[valid_ki[di], :].argmax(axis=0)] for di in range(ND)], dtype=np.int16).T
    policy_keep2[mask] = np.array([valid_ki[di][EV2[valid_ki[di], :].argmax(axis=0)] for di in range(ND)], dtype=np.int16).T

# ── Kategorinavnet ────────────────────────────────────────────────────────────
CAT_NAMES = [
    "Ettor", "Tvåor", "Treor", "Fyror", "Femmor", "Sexor",
    "Ett par", "Två par", "Tretal", "Fyrtal",
    "Liten stege", "Stor stege", "Kåk", "Chans", "Yatzy"
]

# ── Hjälpfunktion: omvandla kappa till mask ───────────────────────────────────
def kappa_to_mask(kappa):
    mask = 0
    for i, k in enumerate(kappa):
        if k == 1:
            mask |= (1 << i)
    return mask

# ── Slå upp optimal handling givet tillstånd ─────────────────────────────────
def optimal_handling(delta, kappa, rho, gamma):
    """
    delta: tuple av längd 6, t.ex. (0,0,0,0,4,1)
    kappa: tuple av längd 15, t.ex. (1,1,1,1,1,1,1,1,1,0,1,1,1,1,1)
    rho:   antal återstående omslag, 0, 1 eller 2
    gamma: poäng kvar till bonus, 0-63
    """
    if delta not in DICE_IDX:
        raise ValueError(f"Ogiltigt tärningsutfall: {delta}")
    di   = DICE_IDX[delta]
    mask = kappa_to_mask(kappa)

    if rho == 0:
        # Kategorival
        cat = int(policy_cat[mask, gamma, di])
        print(f"Tillstånd: delta={delta}, kappa={kappa}, rho={rho}, gamma={gamma}")
        print(f"Optimal handling: välj kategori '{CAT_NAMES[cat]}' (kategori {cat+1})")
        print(f"Poäng: {score_table[di, cat]}")
    elif rho == 1:
        # Andra omslagsbeslutet (policy_keep1)
        ki   = int(policy_keep1[mask, gamma, di])
        keep = ALL_KEEPS[ki]
        print(f"Tillstånd: delta={delta}, kappa={kappa}, rho={rho}, gamma={gamma}")
        print(f"Optimal handling: behåll tärningar {keep}")
        kept_vals = [v+1 for v in range(N_SIDES) for _ in range(keep[v])]
        print(f"Dvs. behåll: {kept_vals}")
    elif rho == 2:
        # Första omslagsbeslutet (policy_keep2)
        ki   = int(policy_keep2[mask, gamma, di])
        keep = ALL_KEEPS[ki]
        print(f"Tillstånd: delta={delta}, kappa={kappa}, rho={rho}, gamma={gamma}")
        print(f"Optimal handling: behåll tärningar {keep}")
        kept_vals = [v+1 for v in range(N_SIDES) for _ in range(keep[v])]
        print(f"Dvs. behåll: {kept_vals}")
    else:
        raise ValueError(f"Ogiltigt rho: {rho}, måste vara 0, 1 eller 2")

# ── Exempel: tillstånd x från beviset ────────────────────────────────────────
# Fyra femmor och en sexa, endast fyrtal öppet, ett kast kvar (rho=1), gamma=0
#print("\n--- Tillstånd x från beviset ---")
# delta = (0, 0, 0, 0, 4, 1)
#kappa = (1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1)  # alla stängda utom fyrtal (index 9)
#rho   = 1
#gamma = 0
#optimal_handling(delta, kappa, rho, gamma)

#  ── Fler tillstånd kan läggas till här ───────────────────────────────────────
# Avkommentera och ändra för att testa andra tillstånd:
print("\n--- Eget tillstånd ---")
optimal_handling(
    delta = (0, 0, 0, 0, 1, 4),
    kappa = (1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1),
    rho   = 0,
    gamma = 22
)

