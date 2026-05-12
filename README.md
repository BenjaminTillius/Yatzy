> ⚠️ **OBS:** Programmen kräver små ändringar i koden för att uppnå önskat resultat. Med "Input" i beskrivngarna menas att t.ex ett tillstånd, en totalpoäng eller Beta måste ändras manuellt i koden, inte att programmet tar någon input. Seeds är satta så att givet att "inputen" är densamma som beskriven i arbetet bör resultatet bli detsamma. 

## 📁 Projektstruktur

### `2-yatzy/`
Innehåller alla filer som beräknar eller simulerar spel av 2-Yatzy, både med π^Max och π^Rekord_Beta strategierna.

| Fil | Beskrivning |
|-----|-------------|
| `2-yatzy_max.py` | Beräknar det förväntade värdet för 2-Yatzy givet π^Max |
| `2-yatzy_max_teori.py` | Sparar π^Max för 2-Yatzy och beräknar förväntade värdet när 2-Yatzy spelas som Rekord-Yatzy |
| `2-yatzy_rekord.py` | Beräknar det förväntade värdet för 2-Yatzy givet π^Rekord_Beta |
| `2-yatzy_rekord_fördelning.py` | Beräknar förväntade värdet för alla Beta |
| `2-yatzy_rekord_teori.py` | Sparar π^Rekord_Beta och beräknar förväntade värdet med originala belöningsfunktioner |

### `Beskrivande versioner/`
Innehåller två av programmen där koden är ändrad för att vara mer beskrivande och förhoppningsvis lättare att läsa.

| Fil | Beskrivning |
|-----|-------------|
| `yatzy_max_beskrivande.py` | Beskrivande version av `yatzy_max.py` |
| `yatzy_rekord_beskrivande.py` | Beskrivande version av `yatzy_rekord.py` |

### `yatzy_max/`
Innehåller filer som beräknar eller simulerar värden för π^Max.

| Fil | Beskrivning |
|-----|-------------|
| `yatzy_max.py` | Beräknar det förväntade värdet för Yatzy givet π^Max |
| `yatzy_max_flerspelare.py` | Simulerar spel med flera spelare som följer π^Max och sparar statistik |
| `yatzy_max_hitta_handling.py` | **Input:** tillstånd → **Output:** handling givet π^Max |
| `yatzy_max_hitta_slutpoäng.py` | **Input:** totalpoäng → **Output:** simulerat spel i det terminaltillståndet givet π^Max |
| `yatzy_max_hitta_tillstånd.py` | **Input:** tillstånd → **Output:** simulerat spel i det tillståndet givet π^Max |
| `yatzy_max_sim.py` | Beräknar förväntade värdet, simulerar x spel och sparar statistik |
| `yatzy_utan_bonus.py` | Beräknar förväntade värdet för Yatzy utan bonus givet π^Max |
| `felaktig_tvåpar.txt` | Kodsnutt som ger förväntat värde 248.6329 om den ersätter `cat==7` i `score` i `yatzy_max.py` |

### `yatzy_rekord/`
Innehåller filer som beräknar eller simulerar värden för π^Rekord_Beta.

> ⚠️ **OBS:** Körtiden är minst 3 timmar för varje program.

| Fil | Beskrivning |
|-----|-------------|
| `yatzy_rekord.py` | Beräknar det förväntade värdet för Yatzy givet π^Rekord_Beta |
| `yatzy_rekord_fördelning.py` | Beräknar förväntade värdet för alla Beta *(374 × minst 3h = för lång körtid)* |
| `yatzy_rekord_hitta_handling.py` | **Input:** tillstånd → **Output:** handling från π^Optimal_Beta |
| `yatzy_rekord_sim.py` | Beräknar förväntade värdet, simulerar x spel och sparar statistik |
