"""
A*-algoritmi ruudukolle, toteutettu yleisen best-first-haun päälle valittavalla heuristiikalla.

A* laajentaa Dijkstraa lisäämällä heuristiikan h(s), joka arvioi jäljellä olevan matkan päätepisteeseen. Tämä ohjaa hakua loppuun ja
vähentää tutkittujen solmujen määrää, kunhan heuristiikka on ylärajaton.
"""

from neighbors import get_neighbors
from search_core import best_first_search

def astar(grid, start, goal, heuristic, allow_diagonal = False):
    """
    Suorittaa A*-algoritmin ruudukolla.

    Parametrit:
        grid (grid.Grid): Ruudukko, jolla haku suoritetaan
        start (tuple[int, int]): Lähtöpisteen koordinaatit (x,y)
        goal (tuple[int, int]): Päätepisteen koordinaatit (x,y)
        heuristic (Callable[[tuple[int, int], tuple[int, int]], float]): Heuristiikkafunktio (piste, piste) -> arvioitu etäisys.
        allow_diagonal (bool): Jos True, sallitaan 8-suuntainen liikkuminen

    Palauttaa:
        search_core.SearchResult, joka sisältää löydetyn polun, sen kustannuksen ja
        hakutilastot (mm. laajennettujen solujen määrä)
    """
    def neighbor_fn(node):
        x, y = node
        return get_neighbors(grid, x, y, allow_diagonal=allow_diagonal)
    
    def heuristic_fn(node):
        return heuristic(node, goal)
    
    return best_first_search(start, goal, neighbor_fn, heuristic_fn=heuristic_fn)