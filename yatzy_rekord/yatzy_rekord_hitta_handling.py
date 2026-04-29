"""
Rekordspel: beräknar sannolikheten att uppnå minst BETA poäng i skandinaviskt
Yatzy med optimal strategi, och slår upp optimal handling givet ett tillstånd.

Indata för optimal_handling_rekord:
  delta: tuple av längd 6, antal tärningar av varje valör (1-6)
  kappa: tuple av längd 15, 0=öppen 1=stängd
  rho:   antal återstående omslag (0, 1 eller 2)
  tau:   totalpoäng hittills (0..BETA)
  gamma: poäng kvar till bonus (0..63)
"""
import numpy as np
from itertools import product as iproduct
from math import factorial

N_DICE, N_SIDES, N_CATS = 5, 6, 15
GAMMA_MAX = 63
NU        = GAMMA_MAX + 1
BONUS     = 50

BETA = 287
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
            nt    = new_tau[cat]
            ng    = new_gamma[cat]
            nt_b  = np.minimum(nt + BONUS, BETA)
            nt_3d  = nt[:, None, :]
            ntb_3d = nt_b[:, None, :]
            ng_3d  = ng[None, :, :]
            gamma_before = np.arange(NU)[None, :, None]
            bonus_trigger = (gamma_before > 0) & (ng_3d == 0)
            new_tau_3d = np.where(bonus_trigger, ntb_3d, nt_3d)
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

# ── Kategorinavnet ────────────────────────────────────────────────────────────
CAT_NAMES = [
    "Ettor", "Tvåor", "Treor", "Fyror", "Femmor", "Sexor",
    "Ett par", "Två par", "Tretal", "Fyrtal",
    "Liten stege", "Stor stege", "Kåk", "Chans", "Yatzy"
]

# ── Hjälpfunktioner ───────────────────────────────────────────────────────────
def kappa_to_mask(kappa):
    mask = 0
    for i, k in enumerate(kappa):
        if k == 1:
            mask |= (1 << i)
    return mask

def next_tau_gamma(cat, di, tau, gamma):
    """Beräkna nytt tau och gamma efter kategorival."""
    r  = int(score_table[di, cat])
    ng = max(0, gamma - r) if cat < 6 else gamma
    nt = min(tau + r, BETA)
    if cat < 6 and gamma > 0 and ng == 0:
        nt = min(nt + BONUS, BETA)
    return nt, ng

# ── Slå upp optimal handling givet tillstånd ─────────────────────────────────
def optimal_handling_rekord(delta, kappa, rho, tau, gamma):
    """
    delta: tuple av längd 6, t.ex. (0,0,0,0,4,1)
    kappa: tuple av längd 15
    rho:   antal återstående omslag (0, 1 eller 2)
    tau:   totalpoäng hittills (0..BETA)
    gamma: poäng kvar till bonus (0..63)
    """
    if delta not in DICE_IDX:
        raise ValueError(f"Ogiltigt tärningsutfall: {delta}")
    di   = DICE_IDX[delta]
    mask = kappa_to_mask(kappa)
    tau  = min(tau, BETA)

    print(f"\nTillstånd: delta={delta}, tau={tau}, gamma={gamma}, rho={rho}")
    
    open_cats = [c for c in range(N_CATS) if not (mask >> c & 1)]

    if rho == 0:
        # Kategorival
        best_cat, best_val = -1, -1.0
        print(f"\nAlla möjliga kategorival:")
        for cat in open_cats:
            nm       = mask | (1 << cat)
            nt, ng   = next_tau_gamma(cat, di, tau, gamma)
            val      = float(round_start[nm, nt, ng])
            print(f"  {CAT_NAMES[cat]:<14}: P(>= {BETA}) = {val:.4f}  (poäng={int(score_table[di,cat])}, ny tau={nt}, ny gamma={ng})")
            if val > best_val:
                best_val = val; best_cat = cat
        print(f"\nOptimal handling: välj kategori '{CAT_NAMES[best_cat]}'")

    else:
        # Omslagsval
        # Beräkna V0[di] = max_cat round_start[nm, new_tau, new_gamma]
        V0 = np.zeros(ND, dtype=np.float32)
        for cat in open_cats:
            for d in range(ND):
                nt, ng = next_tau_gamma(cat, d, tau, gamma)
                nm     = mask | (1 << cat)
                val    = float(round_start[nm, nt, ng])
                if val > V0[d]:
                    V0[d] = val

        if rho == 1:
            ev  = T @ V0
            ki  = valid_ki[di][ev[valid_ki[di]].argmax()]
        else:  # rho == 2
            ev1 = T @ V0
            V1  = np.array([ev1[valid_ki[d]].max() for d in range(ND)])
            ev  = T @ V1
            ki  = valid_ki[di][ev[valid_ki[di]].argmax()]

        keep = ALL_KEEPS[ki]
        kept_vals = [v+1 for v in range(N_SIDES) for _ in range(keep[v])]
        print(f"Optimal handling: behåll tärningar {keep}")
        print(f"Dvs. behåll: {kept_vals}")

# ── Exempel ───────────────────────────────────────────────────────────────────
print("\n--- Tillstånd x ---")
optimal_handling_rekord(
    delta = (0, 0, 0, 0, 4, 1),
    kappa = (1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1),
    rho   = 1,
    tau   = 265,
    gamma = 0
)

