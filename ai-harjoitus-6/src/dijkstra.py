"""
Dijkstran algoritmi ruudukolla, toteutettu yleisen best-first-haun päälle heuristiikalla h(s) = 0.

Dijkstra takaa lyhimmän polun tutkimalla solmuja tasaisesti kaikki suuntiin lähtöpisteestä,
ilman "ideaa" siitä, missä päätepiste on. Sen takia Dijkstran algoritmi on hitaampi kuin A* (tutkii enemmän solmuja),
mutta se hyvä tilanteissa, joissa (hyvää) heuristiikkaa ei ole saatavilla.
"""

from neighbors import get_neighbors
from search_core import best_first_search

def dijkstra(grid, start, goal, allow_diagonal = False):
    """
    Suorittaa Dijkstran algoritmin ruudukolla.

    Parametrit:
        grid (grid.Grid): Ruudukko, jolla hakusuoritetaan
        start (tuple[int, int]): Lähtöpisteen koordinaatit (x,y)
        goal (tuple[int, int]): Päätepisteen koordinaatit (x,y)
        allow_diagonal (bool): Jos True, sallitaan 8-suuntainen liikkuminen.

    Palauttaa:
        search_core.SearchResult, joka sisältää löydetyn polun, sen kustannnuksen ja
        hakutilastot (mm. laajennettujen solujen määrä)
    """
    def neighbor_fn(node):
        x, y = node
        return get_neighbors(grid, x, y, allow_diagonal=allow_diagonal)
    
    # Dijkstra eli best-first-haku ilman heuristiikka (h(s) = 0 kaikille s)
    return best_first_search(start, goal, neighbor_fn, heuristic_fn=None)