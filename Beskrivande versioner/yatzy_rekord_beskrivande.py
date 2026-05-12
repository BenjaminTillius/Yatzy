"""
Rekordspel: beräknar sannolikheten att uppnå minst BETA poäng i skandinaviskt
Yatzy med optimal strategi, med hjälp av bakåtinduktion.

Bonus: +50 poäng om övre sektionen (ettor–sexor) ger minst 63 poäng.
Bonus läggs bara till när gamma övergår från >0 till 0 (dvs. när bonusen
faktiskt uppnås), inte varje gång en övre kategori stängs när bonus redan är säkrad.

Tillstånd: (kategorimask, totalpoäng, gamma, tärningsutfall, omslag)
  totalpoäng: poäng hittills, cappat vid BETA
  gamma:      poäng kvar till bonus (0..63), gamma=0 betyder att bonus är säkrad
  kategorimask: 15-bitars heltal där bit c = 1 om kategori c är stängd

rundstart[kategorimask, totalpoäng, gamma] = sannolikheten att uppnå BETA poäng
sparas löpande (~2.5 GB för BETA=300).
"""
import numpy as np
from itertools import product as iproduct
from math import factorial

# ── Spelets grundparametrar ───────────────────────────────────────────────────
antal_tärningar   = 5   # n i uppsatsen
antal_sidor       = 6   # k i uppsatsen
antal_kategorier  = 15  # h i uppsatsen
bonusgräns        = 63  # gamma_max i uppsatsen: poäng i övre sektionen för bonus
antal_gammavärden = bonusgräns + 1  # gamma kan vara 0, 1, ..., 63
bonuspoäng        = 50  # B i uppsatsen

# ── Rekordmålet ───────────────────────────────────────────────────────────────
# BETA är den poäng spelaren strävar efter att uppnå.
# Alla totalpoäng >= BETA behandlas som BETA (cappning).
BETA             = 330        # ← ändra detta för att beräkna andra rekordmål
antal_poängvärden = BETA + 1  # totalpoäng kan vara 0, 1, ..., BETA

# ── Multinomialkoefficienten ──────────────────────────────────────────────────
# Beräknar n! / (delta_1! * delta_2! * ... * delta_k!)
# Anger på hur många sätt man kan ordna n tärningar med delta_i av värde i.
# Används för att beräkna sannolikheter för tärningsutfall.
def multinomialkoefficient(tärningar):
    n = sum(tärningar)
    resultat = factorial(n)
    for antal in tärningar:
        resultat //= factorial(antal)
    return resultat

# ── Alla möjliga tärningsutfall och keeps ─────────────────────────────────────
# Ett tärningsutfall representeras som (delta_1,...,delta_6) där delta_i är
# antalet tärningar med värde i. Summan är alltid exakt antal_tärningar (=5).
# Ett keep är samma format men summan får vara 0 till 5 — de tärningar man behåller.

alla_utfall  = [u for u in iproduct(range(antal_tärningar+1), repeat=antal_sidor)
                if sum(u) == antal_tärningar]   # 252 möjliga utfall

alla_keeps   = [k for k in iproduct(range(antal_tärningar+1), repeat=antal_sidor)
                if sum(k) <= antal_tärningar]   # 462 möjliga keeps

utfall_index = {u:i for i,u in enumerate(alla_utfall)}  # utfall -> index

antal_utfall = len(alla_utfall)  # 252
antal_keeps  = len(alla_keeps)   # 462

utfall_matris = np.array(alla_utfall, dtype=np.int8)  # (252, 6)
keeps_matris  = np.array(alla_keeps,  dtype=np.int8)  # (462, 6)

# ── Övergångsmatrisen ─────────────────────────────────────────────────────────
# övergång[ki, di] = sannolikheten att hamna i tärningsutfall di
#                    om man behåller keep ki och kastar om resten.
# Beräknas från övergångsfunktionen p_t(j | s_t, a) i uppsatsen.
övergång = np.zeros((antal_keeps, antal_utfall), dtype=np.float32)
for ki, keep in enumerate(alla_keeps):
    antal_omslagda       = antal_tärningar - sum(keep)
    totalt_antal_utfall  = antal_sidor ** antal_omslagda
    for fria in iproduct(range(antal_omslagda+1), repeat=antal_sidor):
        if sum(fria) != antal_omslagda:
            continue
        resultat = tuple(keep[i]+fria[i] for i in range(antal_sidor))
        if resultat in utfall_index:
            övergång[ki, utfall_index[resultat]] += multinomialkoefficient(fria) / totalt_antal_utfall

# Sannolikheter för varje utfall vid rundestart (kasta alla 5 tärningar)
rundstart_sannolikheter = np.array([multinomialkoefficient(u) / antal_sidor**antal_tärningar
                                    for u in alla_utfall], dtype=np.float32)

# ── Giltiga keeps per tärningsutfall ─────────────────────────────────────────
# giltiga_keeps[di] = lista med index till keeps som är giltiga för utfall di,
# dvs. keeps där man inte behåller fler tärningar av ett värde än man slagit.
giltiga_keeps = [np.where(np.all(keeps_matris <= utfall_matris[di], axis=1))[0]
                 for di in range(antal_utfall)]

# ── Poängfunktionen ───────────────────────────────────────────────────────────
# Beräknar poängen för ett tärningsutfall i en given kategori.
# Kategori 0-5: ettor–sexor (övre sektionen)
# Kategori 6-14: ett par, två par, tre lika, fyra lika,
#                liten stege, stor stege, kåk, chans, Yatzy
def poäng(tärningsutfall, kategori):
    c = list(tärningsutfall)
    if kategori <= 5:
        return c[kategori] * (kategori + 1)
    if kategori == 6:   # ett par: högsta paret
        for v in range(5,-1,-1):
            if c[v] >= 2: return 2*(v+1)
        return 0
    if kategori == 7:   # två par: de två högsta paren
        par = [v+1 for v in range(6) if c[v] >= 2]
        return 2*sum(sorted(par)[-2:]) if len(par) >= 2 else 0
    if kategori == 8:   # tre lika: högsta tretal
        for v in range(5,-1,-1):
            if c[v] >= 3: return 3*(v+1)
        return 0
    if kategori == 9:   # fyra lika: högsta fyrtal
        for v in range(5,-1,-1):
            if c[v] >= 4: return 4*(v+1)
        return 0
    if kategori == 10:  # liten stege: 1-2-3-4-5
        return 15 if all(c[v] >= 1 for v in range(5)) else 0
    if kategori == 11:  # stor stege: 2-3-4-5-6
        return 20 if all(c[v] >= 1 for v in range(1,6)) else 0
    if kategori == 12:  # kåk: tre lika + par av olika värden
        tretal = next((v+1 for v in range(5,-1,-1) if c[v] >= 3), None)
        par    = next((v+1 for v in range(5,-1,-1) if c[v] >= 2 and (v+1) != tretal), None)
        return 3*tretal + 2*par if tretal and par else 0
    if kategori == 13:  # chans: summan av alla tärningar
        return sum((v+1)*c[v] for v in range(6))
    if kategori == 14:  # Yatzy: fem lika
        return 50 if any(x == 5 for x in c) else 0
    return 0

# Förberäknad poängtabell: poäng_tabell[di, kat] = poäng för utfall di i kategori kat
poäng_tabell = np.array([[poäng(u, kat) for kat in range(antal_kategorier)]
                          for u in alla_utfall], dtype=np.float32)  # (252, 15)

# ── Ny totalpoäng och nytt gamma efter kategorival ────────────────────────────
# ny_totalpoäng[kat][tau, di] = min(tau + poäng[di, kat], BETA)
# nytt_gamma[kat][gamma, di]  = max(0, gamma - poäng[di, kat])
# ny_totalpoäng beräknas för alla kategorier, nytt_gamma bara för övre (0-5).

totalpoäng_vektor = np.arange(antal_poängvärden)[:, None]   # (NT, 1)
ny_totalpoäng = {kat: np.minimum(totalpoäng_vektor + poäng_tabell[:,kat][None,:], BETA).astype(np.int32)
                 for kat in range(antal_kategorier)}         # (NT, 252) per kategori

gamma_vektor = np.arange(antal_gammavärden)[:, None]        # (64, 1)
nytt_gamma   = {kat: np.maximum(0, gamma_vektor - poäng_tabell[:,kat][None,:]).astype(np.int32)
                for kat in range(6)}                         # (64, 252) per kategori

# ── Bästa omslagsval ─────────────────────────────────────────────────────────
# Tar in V (sannolikhet att nå BETA per totalpoäng, gamma och utfall) och
# returnerar det bästa möjliga värdet efter ett omslag.
# V har formen (antal_poängvärden * antal_gammavärden, antal_utfall) — upplattad.
def bästa_omslag(V):
    """V: (NT*NU, 252) → V_nästa: (NT*NU, 252)"""
    förväntat_värde = övergång @ V.T                          # (462, NT*NU)
    V_nästa = np.empty_like(V)
    for di in range(antal_utfall):
        V_nästa[:, di] = förväntat_värde[giltiga_keeps[di], :].max(axis=0)
    return V_nästa

# ── Rundstart-tabellen ────────────────────────────────────────────────────────
# rundstart[kategorimask, totalpoäng, gamma] = sannolikheten att uppnå BETA poäng
# från och med rundestart, givet kategorimask, totalpoäng och gamma.
antal_maskar = 1 << antal_kategorier   # 2^15 = 32768
full_mask    = antal_maskar - 1        # alla kategorier stängda

print(f"Allokerar rundstart ({antal_maskar}×{antal_poängvärden}×{antal_gammavärden}, "
      f"{antal_maskar*antal_poängvärden*antal_gammavärden*4/1024**3:.2f} GB)...")
rundstart = np.zeros((antal_maskar, antal_poängvärden, antal_gammavärden), dtype=np.float32)

# Terminalbelöning: r_N = 1 om totalpoäng = BETA (dvs totalpoäng >= BETA i cappat tillstånd)
rundstart[full_mask, BETA, :] = 1.0

# ── Bakåtinduktion ────────────────────────────────────────────────────────────
# Itererar bakåt från nästan-terminal mot starttillståndet.
# För varje kategorimask beräknas sannolikheten att nå BETA vid rundestart.
print(f"Bakåtinduktion för BETA={BETA}...")
for kategorimask in range(full_mask - 1, -1, -1):
    öppna_kategorier = [k for k in range(antal_kategorier)
                        if not (kategorimask >> k & 1)]
    if not öppna_kategorier:
        continue

    # Kategorival: V0[totalpoäng, gamma, di] = max över öppna kategorier av
    # { rundstart[ny_mask, ny_totalpoäng, nytt_gamma] }
    # Belöningen r_t = 0 för alla handlingar (rekordspel), därav ingen poängterm.
    V0 = np.zeros((antal_poängvärden, antal_gammavärden, antal_utfall), dtype=np.float32)

    for kat in öppna_kategorier:
        ny_mask   = kategorimask | (1 << kat)
        rs_ny_mask = rundstart[ny_mask]                    # (NT, NU)

        if kat < 6:
            # Övre kategori: både totalpoäng och gamma uppdateras.
            # Bonus läggs till i totalpoängen när gamma övergår från >0 till 0.
            ny_poäng_utan_bonus = ny_totalpoäng[kat]       # (NT, ND)
            ny_gammavärde       = nytt_gamma[kat]          # (NU, ND)
            ny_poäng_med_bonus  = np.minimum(ny_poäng_utan_bonus + bonuspoäng, BETA)  # (NT, ND)

            # Expandera till tre dimensioner (totalpoäng, gamma, utfall)
            # för att kunna indexera rundstart[ny_mask] som har samma form.
            ny_poäng_utan_bonus_3d = ny_poäng_utan_bonus[:, None, :]   # (NT, 1,  ND)
            ny_poäng_med_bonus_3d  = ny_poäng_med_bonus[:, None, :]    # (NT, 1,  ND)
            ny_gammavärde_3d       = ny_gammavärde[None, :, :]         # (1,  NU, ND)
            gamma_före_kategorival = np.arange(antal_gammavärden)[None, :, None]  # (1, NU, 1)

            # Bonus triggas bara när gamma övergår från >0 till 0
            bonus_utlöses    = (gamma_före_kategorival > 0) & (ny_gammavärde_3d == 0)
            ny_totalpoäng_3d = np.where(bonus_utlöses,
                                        ny_poäng_med_bonus_3d,
                                        ny_poäng_utan_bonus_3d)        # (NT, NU, ND)

            rs = rs_ny_mask[ny_totalpoäng_3d,
                            ny_gammavärde_3d * np.ones((antal_poängvärden, 1, 1), dtype=np.int32)]
        else:
            # Undre kategori: bara totalpoäng uppdateras, gamma oförändrat.
            ny_poäng = ny_totalpoäng[kat]                  # (NT, ND)
            rs = rs_ny_mask[ny_poäng[:, None, :],
                            np.arange(antal_gammavärden)[None, :, None]]  # (NT, NU, ND)

        np.maximum(V0, rs, out=V0)

    # Omslagsval: två omslag per runda
    # Platta ut (totalpoäng, gamma) till en dimension för matrisoperationen
    V0_flat = V0.reshape(antal_poängvärden * antal_gammavärden, antal_utfall)
    V1_flat = bästa_omslag(V0_flat)   # efter 1 omslag
    V2_flat = bästa_omslag(V1_flat)   # efter 2 omslag

    # Spara sannolikheten att nå BETA vid rundestart för denna mask
    rundstart[kategorimask] = (rundstart_sannolikheter @ V2_flat.T).reshape(
        antal_poängvärden, antal_gammavärden)

# ── Resultat ──────────────────────────────────────────────────────────────────
# Starttillståndet: inga kategorier stängda (mask=0),
# totalpoäng=0, gamma=63 (ingen bonus ännu)
print(f"\nu*(s_1) = P(>= {BETA} poäng | optimal strategi) = {rundstart[0, 0, bonusgräns]:.4f}")
