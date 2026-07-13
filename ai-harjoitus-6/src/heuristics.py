"""
Heuristiikkafunktio A*-algoritmia varten.

Heuristiikka h(s) arvioi jäljellä olevan kustannuksen solmusta s päätepisteeseen.
Jotta A* takaisi optimaalisen polun, heuristiikan on oltava ylärajaton ja kelvollinen

Ylärajattomuus riippuu siitä, mitkä siirtymät liikkumismalli sallii:

  - Manhattan-etäisyys (|dx| + |dy|):
    Kelvollinen vain 4-suuntaisessa liikkumisessa. 8-suuntaisessa liikkumisessa Manhattan
    yliarvioi kustannuksen aina, kun optimaalinen plku käyttää diagonaaleja. ÄLÄ käytä Manhattania
    8-suuntaisessa liikkumisessa, jos optimaalisuus on vaatimus.

  - Euklidinen etäisyys (sqrt(dx^2 + dy^2)):
    Kelvollinen sekä 4- että 8-suuntaisessa liikkumisessa, koska suora viiva on aina lyhin mahdollinen
    matka pisteestä toiseen. Se kuitenkin aliarvioi 8-suuntaisessa liikkumisessa enemmän kuin Octile,
    joten se on vähemmän informoiva. Tutkii enemmän solmuja kuin Octile.

  - Chebyshev-etäisyys (max(|dx|, |dy|)):
    Kelvollinen vain 8-suuntaisessa liikkumisesas, jos diagonaalin hinta on 1 (ei sqrt(2)). Tässä
    projektissa diagonaalit maksavat sqrt(2), joten periaatteessa perus-Chebyshev ei ole kelvollinen tässä liikkumismallisa.
    Kuitenkin tässä projektissa Chebyshev on teknisesti kelvollinen (koska se aliarvioi), mutta se ei ole tiukka heuristiikka
    sqrt(2)-diagonaalimallissa. Käytetään lähinnä vertailukohteena.

  - Octile-etäisyys:
    Kelvollinen 8-suuntaisessa liikkumisessa, kun diagonaalin hinta on juuri sqrt(2). Se on tiukka heuristiikka
    tälle liikkumismallille. Se laskee tarkalleen, kuinka monta diagonaali- ja suora-askelta lyhin mahdollinen
    reitti esteettömässä ruudukossa vaatisi. Suositeltu heuristiikka 8-suuntaiselle liikkumiselle tässä projektissa.

4-suuntainen: Manhattan (tiukka), Euklidinen (kelvollinen, mutta löysempi)
8-suuntainen (diagonaali = sqrt(2)): Octile (tiukka), Euklidinen (kelvollinen), Chebysheb (kelvollinen, mutta hyvin löysä)
8-suuntainen EI-SUOSITELTU: Manhattan (epäkelvollinen, voi antaa epäoptimaalisia polkuja)
"""

from math import sqrt

def manhattan(a, b):
    """
    Manhattan-etäisyys (L1-normi).
    Kelvollinen vain 4-suuntaisessa liikkumisessa. Ei suositella 8-suuntaiseen liikkumiseen.

    Parametrit:
        a (tuple[int, int]): Ensimmäinen piste (x,y)
        b (tuple[int, int]): Toinen piste (x,y)

    Palauttaa:
        |ax - bx| + |ay - by|
    """
    (ax, ay), (bx, by) = a, b
    return abs(ax - bx) + abs(ay - by)

def euclidean(a, b):
    """
    Euklidinen etäisyys (L2-normi, suora viiva pisteiden välillä).
    Kelvollinen sekä 4- että 8-suuntaisesas liikkumisessa, koska suora viiva on aina
    lyhin mahdollinen matka.

    Parametrit:
        a (tuple[int, int]): Ensimmäinen piste
        b (tuple[int, int]): Toinen piste

    Palauttaa:
        sqrt((ax - bx)^2 + (ay - by)^2)
    """
    (ax, ay), (bx, by) = a, b
    return sqrt((ax - bx) ** 2 + (ay - by) ** 2)

def chebyshev(a, b):
    """
    Chebyshev-etäisyys (L-infinity-normi). Kelvollinen 8-suuntaisessa liikkumisessa vain,
    jos diagonaalin hinta on 1.

    Parametrit:
        a (tuple[int, int]): Ensimmäinen piste
        b (tuple[int, int]): Toinen piste

    Palauttaa:
        max(|ax - bx|, |ay - by|)
    """
    (ax, ay) = (bx, by) = a, b
    return max(abs(ax - bx), abs(ay - by))

def octile(a, b):
    """
    Octile-etäisyys: Tiukka kelvollinen heuristiikka 8-suuntaiselle liikkumiselle,
    kun diagonaalin hinta on sqrt(2) ja suoran askeleen hinta on 1.
    Ei sovellu sellaisenaan 4-suuntaiseen liikkumiseen.

    Parametrit:
        a (tuple[int, int]): Ensimmäinen piste
        b (tuple[int, int]): Toinen piste

    
    Palauttaa:
        min(dx, dy) * sqrt(2) + |dx - dy| * 1, missä dx = |ax - bx| ja dy = |ay - by|
    """
    (ax, ay), (bx, by) = a, b
    dx = abs(ax - bx)
    dy = abs(ay - by)
    return (dx + dy) + (sqrt(2 - 2) * min(dx, dy))

def zero(_a, _b):
    """
    Nollaheuristiikka: h(s) = 0 kaikille solmuille.
    Heuristiikka, joka muuttaa yleisen best-first-haun Dijkstran algoritmiksi.
    Triviaalisti kelvellinen.

    Parametrit:
      _a (tuple[int, int]): Ensimmäinen piste, ei käytetä
      _b (tuple[int, int]): Toinen piste, ei käytetä

    Palauttaa: 0.0
    """
    return 0.0

HEURISTICS = {
    "manhattan": manhattan,
    "euklidinen": euclidean,
    "chebyshev": chebyshev,
    "octile": octile,
    "nolla": zero
}