Innehållsförteckning:

2-yatzy: Innehåller alla filer som beräknar eller simulerar spel av 2-Yatzy, både med pi^Max och pi^Rekord_Beta strategierna
        - 2-yatzy_max.py: Beräknar det förväntade värdet för 2-Yatzy givet pi^Max
        - 2-yatzy_max_teori.py: Sparar pi^Max för för 2-Yatzy och beräknar sedan förväntade värdet när 2-Yatzy spelas som Rekord-Yatzy
        - 2-yatzy_rekord.py: Beräknar det förväntade värdet för 2-Yatzy givet pi^Rekord_Beta 
        - 2-yatzy_rekord_fördelning.py: Beräknar det förväntade värdet för 2-Yatzy givet pi^Rekord_Beta för alla Beta
        - 2-yatzy_rekord_teori.py: Sparar pi^Rekord_Beta för 2-Yatzy och beräknar sedan förväntade värdet när 2-Yatzy spelas med orginal belöningsfunktioner

yatzy_max: Innehåller filer som beräknar eller simulerar värden för pi^Max
        - felaktig_tvåpar.txt: En del av kod som ifall den ersätter cat==7 i score i yatzy_max.py ger förväntade värdet 248.6329.
        - yatzy_max.py: Beräknar det förväntade värdet för Yatzy givet pi^Max
        - yatzy_max_flerspelare.py: Simulerar spel med flera spelare där alla följer pi^Max och sparar statistik för vinnaren
        - yatzy_max_hitta_handling.py: Input: Ett tillstånd i Yatzy, Output: Handling framtagen av pi^Max
        - yatzy_max_hitta_slutpoäng.py: Input: Totalpoängen i ett terminaltillstånd, Output: Ett simulerat spel som hamnat i det terminaltillståndet
        - yatzy_max_hitta_tillstånd.py: Input: Ett tillstånd i Yatzy, Output: Ett simulerat spel som hamnat i det tillståndet
        - yatzy_max_sim.py: Beräknar det förväntade värdet för Yatzy givet pi^Max, simulerar x antal spel och sparar statistik
        - yatzy_utan_bonus.py: Beräknar det förväntade värdet för Yatzy utan bonus givet pi^Max

yatzy_rekord: Innehåller filer som beräknar eller simulerar värden för pi^Rekord_Beta, OBS körtiden är minst 3 timmar för varje program
        - yatzy_rekord.py: Beräknar det förväntade värdet för Yatzy givet pi^Rekord_Beta
        - yatzy_rekord.py: Beräknar det förväntade värdet för Yatzy givet pi^Rekord_Beta för alla Beta, OBS för lång körtid (374 x minst 3h)
        - yatzy_rekord_hitta_handling.py: Ett tillstånd i Yatzy, Output: Handling framtagen av pi^Optimal_Beta
        - yatzy_rekord_sim.py: Beräknar det förväntade värdet för Yatzy givet pi^Rekord_Beta, simulerar x antal spel och sparar statistik
        
        
