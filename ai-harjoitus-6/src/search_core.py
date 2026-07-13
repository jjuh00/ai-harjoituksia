"""
Yleinen "best-first search", jota sekä Dijkstra että A* käyttävät.

Suunnitteluhuomiot: Dijkstran algoritmi on vain A* nollaheuristiikaalla.
Molemmat algoritmit ylläpitävät samaa prioriteettijonoa, jossa solmuja käsitellään
pienimmän f(s)-arvon mukaan, missä:

f(s) = g(s) + h(s)
g(s) = todellinen (tunnettu) kustannus lähtopisteestä solmuun s
h(s) = heuristinen arvio solmusta s maaliin (Dijkastralla h(s) = 0 kaikille s, jolloin 
        f(s) = g(s) ja haku etenee tasaisesti joka suuntaan)

Pythonin heapq-pohjainen PriorityQueue ei tue "decrease-key"-opraatiota eli solmun prioriteetin
päivittämistä paikan päällä tehokkaasti. Sen sijaan, että
solumn prioriteettia yritettäisiin päivittää jonossa, kum parempi reitti löytyy,
lisätään vain uusi merkintä. Kun jonosta poistetaan merkintä, tarkistetaan,
onko se yhä ajantasalla (eli vastaako sen g-arvo parasta tunnettua g-arvoa). Jos ei,
merkintä on ohitetaan.
"""

import itertools
import heapq
from typing import TypeVar, Hashable, Callable, Generic
from dataclasses import dataclass

Node = TypeVar("Node", bound=Hashable)

# Naapurifunkio: solmu -> lista (naapuri, siirtymäkustannus) -pareja
NeighborFn = Callable[[Node], list[tuple[Node, float]]]

# Heuristiikka funktio: solmu -> arvioitu kustannus päätepisteeseen
HeuristicFn = Callable[[Node], float]

@dataclass
class SearchResult(Generic[Node]):
    """
    Haun tulos.

    Attribuutit ja parametrit:
        path (list[Node] | None): Lista solmuja alkupisteestä päätepisteeseen (mukaan lukien molemmat päät) tai None, jos polkua ei löytynyt
        total_cost (float | None): Polkun kokonaiskustannus tai None, jos polkua ei löytynyt
        nodes_expanded (int): Montako solmua algoritmi laajensi (posti jonosta ja käsitteli sen naapurit). Mittari A*:n ja Dijkstran vertailussa
        nodes_visited (int): Montako eri solmua jonoon on lisätty
        came_from (dict[Node, Node]): Sanakirja, joka kuvaa jokaisen käsitellyn solmun sen edeltäjään parhaalla löydetyllä polulla
        g_score (dict[Node, float]): Sanakirja parhaista tunnetuista kustannuksista alkupisteestä kuhunkin solmuun
    """
    path: list[Node] | None
    total_cost: float | None
    nodes_expanded: int
    nodes_visited: int
    came_from: dict[Node, Node]
    g_score: dict[Node, float]

def _reconstruct_path(came_from, start, goal):
    """
    Rakentaa polun came_from-sanakirjasta takaperin päätepisteestä alkupisteeseen.

    Parametrit:
        came_from (dict[Node, Node]): Sanakirja solmu -> edeltäjäsolmu
        start (Node): Alkusolmu
        goal (Node): Päätesolmu

    Palauttaa:
        Lista solmusta alkupisteestä päätepisteeseen, molemmat päät mukaan lukien
    """
    path = [goal]
    current = goal
    while current != start:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def best_first_search(start, goal, neighbor_fn, heuristic_fn = None):
    """
    Yleinen best-first-haku, joka toimii sekä Dijkstran algoritmina että A*:na riippuen
    annutusta heuristiikasta.

    Jos heuristic_fn on None (tai palauttaa 0), algoritmi käyttäytyy kuin Dijkstra:
    se laajenee tasaisesti joka suuntaan alkupisteestä. Jos heuristic_fn on kelvollinen
    (eikä koskaan yliarvioi todellista jäljellä olevaa kustannusta), algoritmi käyttäytyy kuin A* ja
    löytää yhä optimaalisen polun, mutta laajentaen tyypillisesti vähemmän solmuja.

    Käyttää laiskaa poistamista prioriteettijonossa, koska Pythonin heapq ei tue decrease-key-operaatiota.

    Parametrit:
        start (Node): Alkusolmu
        goal (Node): Päätesolmu
        neighbor_fn (NeighborFn): Funktio, joka palauttaa solmun naapurit ja siirtymäkustannukset
        heuristic_fn (HeuristicFn | None): Funktio solmu -> arvioitu kustannus päätepisteeseen. None vastaa Dijkstraa (h(s) = 0 kaikille s)

    Palauttaa:
        SearchResult, joka sisältää löydetyn polun tai None
    """
    if heuristic_fn is None:
        heuristic_fn = lambda _n: 0.0 # Dijkstra: nollaheuristiikka

    # Prioriteettijono sisältää (f_score, tiebreak_counter, node)-kolmikkoja.
    # tiebreaker_counter varmistaa vakaan järjestyksen, kun kahdella solmulla on sama
    # f_score, eikä Pythonin heapq yritä vertailla itse solmuja
    counter = itertools.count()
    open_heap = []

    g_score = {start: 0.0}
    came_from = {}

    start_f = heuristic_fn(start)
    heapq.heappush(open_heap, (start_f, next(counter), start))

    # Solmut, joiden optimaalinen g-arvo on jo varmistettu
    closed = set()

    nodes_expanded = 0

    while open_heap:
        _, _, current = heapq.heappop(open_heap)

        # Laiska poisto: jos tämä erkintä on vanhentunut, ohitetaan se
        if current in closed:
            continue

        closed.add(current)
        nodes_expanded += 1

        if current == goal:
            path = _reconstruct_path(came_from, start, goal)
            return SearchResult(
                path=path,
                total_cost=g_score[goal],
                nodes_expanded=nodes_expanded,
                nodes_visited=len(g_score),
                came_from=came_from,
                g_score=g_score
            )
        
        current_g = g_score[current]

        for neighbor, move_cost in neighbor_fn(current):
            if neighbor in closed:
                continue

            tentative_g = current_g + move_cost

            if tentative_g < g_score.get(neighbor, float("inf")):
                # Parempi reitti löytyi: tallennetaan uusi g-arvo ja lisätään uusi merkintä
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f_score = tentative_g + heuristic_fn(neighbor)
                heapq.heappush(open_heap, (f_score, next(counter), neighbor))

    # Polkua ei ole olemassa: jono tyhjeni tai maalia ei löytynyt
    return SearchResult(
        path=None,
        total_cost=None,
        nodes_expanded=nodes_expanded,
        nodes_visited=len(g_score),
        came_from=came_from,
        g_score=g_score
    )