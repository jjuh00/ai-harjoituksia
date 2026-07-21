"""
Kaksisuuntainen haku.

Ideana on suorittaa kaksi hakua samanaikaisesti: yksi etenee eteenpäin alkupisteestä ja
toinen taaksepäin päätepisteestä. Kun molemmat haut ovat käsitelleet saman polun, voidaan
yhdistää kaksi osapolkua yhdeksi kokonaispoluksi. Usein tämä tutkii vähemmän solmuja kuin
yksisuuntaienn haku.

Koska tässä projektissa ruudukossa siirtymisen kustannus riippuu kohdesolmun maastopainosta, 
ruudukko on suunnattu painotettu verkko, vaikka liikkuminen onkin geometrisesti symmetristä.
Tästä syystä taaksepäinhaku ei voi käyttää samaa naapurifunktiota, vaan sille on
annettava erillinen funktio, joka laskee kustannuksen oikeasta suunnasta. Jos ko. funktio jätetään pois
(eli se on None) ja verkko on suuntaamaton, sama naapurifunktio kelpaa molempiin suuntiin.

Toteutus takaa optimaalisen (lyhyimmän) polun, kun molemmat käyttäjävät nollaheuristiikka, eli kaksisuuntaista
Dijkstraa, kunhan taaksepäinhaun naapurifunktio on oikea. Tällöin pysäytysehto on yksinkertainen:
kun pienempien arvojen g-summa on molempien jonojen kärjissä (top_f + top_b) on vähintään paras tähän mennessä
löydetty "silta"-kustannus (mu), mikään vielä laajentamatin solmu ei voi enää parantaa tulostaan.

Funktio hyväksyy myös heuristiikkafuntkiot (kaksisuuntainen A*). Tässä tapauksessa heuristiikka ohjaa hakua
tehokkaasti kohti keskikohtaa, mutta optimaalisuutta ei ole enää yleisesti taattu: itsenäisten eteenpäin- ja taaksepäin-
heuristiikkojen yhdistäminen kaksisuuntaisessa A*:ssa vaatii yhdistetyn potentiaalifunktion (esim. A Fast Algorithm For Finding Better Routes
By AI Search Techniques, Ikeda et al. 1994), mutta se ei ole tämän projektin laajuuden sisällä.
"""

import heapq
import itertools
from dataclasses import dataclass
from typing import TypeVar, Hashable, Callable

Node = TypeVar("Node", bound=Hashable)
NeighborFn = Callable[[Node], list[tuple[Node, float]]]
HeuristicFn = Callable[[Node], float]

@dataclass
class BidirectionalResult:
    """
    Kaksisuuntaisen haun tulos.

    Attribuutit ja parametrit:
        path (list[Node] | None): Löydetty polku alkupisteestä päätepisteeseen tai None
        total_cost (float | None): Polun kokonaiskustannus tai None
        nodes_expanded (int): Molempien suuntien yheenlaskettu laajennettujen solmujen määrä
        meeting_node (Node | Node): Solmu, jossa kaksi hakua kohtasivat, tai None
    """
    path: list[Node] | None
    total_cost: float | None
    nodes_expanded: int
    meeting_node: Node | None

def _expand_one(
    heap, counter, g_score, came_from, closed,
    neighbor_fn, heuristic_fn, other_g_score, update_mu
):
    """
    Laajentaa yhden solmun toisesta hakusuunnasta.

    Parametrit:
        heap (list[tuple[float, int, Node]]): Kyseinen suunnan prioriteettijono
        counter (itertools.count): Jaettu laskuri tasatilanteiden ratkaisuun
        g_score (dict[Node, float]): Kyseisen suunnan g-arvot
        came_from (dict[Node, Node]): Kyseisen suunnan edeltäjätaulukko
        closed (set[Node]): Kyseisen suunnan suljettujen solmujen joukko
        neighbor_fn (NeighborFn): Naapurifunktio tälle suunnalle
        heuristic_fn (HeuristicFn): Heuristiikkafunktio tälle suunnalle
        other_g_score (dict[Node, float]): Vastakkaisen suunnan g-arvot
        update_mu (Callable[[Node, float], None]): Kutsuttava funtio, joka päivittää ulomman mu-muuttujan

    Palauttaa:
        Juuri laajennettu solmu, tai None jos jono oli tyhjä
    """
    while heap:
        _, _, current = heapq.heappop(heap)
        if current in closed:
            continue
        closed.add(current)

        current_g = g_score[current]
        for neighbor, move_cost in neighbor_fn(current):
            tentative_g = current_g + move_cost

            if neighbor not in closed and tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f = tentative_g + heuristic_fn(neighbor)
                heapq.heappush(heap, (f, next(counter), neighbor))

        # 'current' on juuri suljettu tässä suunnassa, joten current_g on lopullinen tästä suunnasta.
        # Jos toinen suunta on jo tavoittanut 'current'-arvon (löytyy sen g-arvoista), summa on aina
        # validi yläraja täydelliselle polulle tämän solmun kautta
        if current in other_g_score:
            update_mu(current, current_g + other_g_score[current])

        return current
    
    return None

def bidirectional_search(
    start, goal, neighbor_fn, heuristic_fn = None,
    reverse_neighbor_fn = None, reverse_heuristic_fn = None
):
    """
    Suorittaa kaksisuuntaisen best-first-haun (Dijkstra tai A* riippuen heuristiikkafunktiosta).

    Parametrit:
        start (Node): Alkusolmu
        goal (Node): Päätesolmu
        neighbor_fn (NeighborFn): Naapurifunktio eteenpäin-suuntaan (start -> goal)
        heuristic_fn (HeuristicFn | None): Heuristiikka solmu -> arvio etäisyydestä päätepisteeseen. None = Dijkstra
        reverse_neighbor_fn (NeighborFn | None): Naapurifunktio taaksepäin-suuntaan (goal -> start). Jos None, oletetaan sama kuin neighbor_fn
        reverse_heuristic_fn (HeuristicFn | None): Heuristiikka solmu -> arvio etäideestä alkuun. None = Dijkstra

    Palauttaa:
        BidirectionalResult, joka sisältää löydetyn polun, kustannuksen, laajennettune solmujen
        kokonaismäärän ja kohtaamissolmun
    """
    if reverse_neighbor_fn is None:
        reverse_neighbor_fn = neighbor_fn
    if heuristic_fn is None:
        heuristic_fn = lambda _n: 0.0
    if reverse_heuristic_fn is None:
        reverse_heuristic_fn = lambda _n: 0.0

    counter = itertools.count()

    g_f = {start: 0.0}
    came_from_f = {}
    closed_f = set()
    heap_f = []
    heapq.heappush(heap_f, (heuristic_fn(start), next(counter), start))

    g_b = {goal: 0.0}
    came_from_b = {}
    closed_b = set()
    heap_b = []
    heapq.heappush(heap_b, (reverse_heuristic_fn(goal), next(counter), goal))

    nodes_expanded = 0

    if start == goal:
        return BidirectionalResult(path=[start], total_cost=0.0, nodes_expanded=0, meeting_node=start)
    
    # mu = paras tähän mennessä löydetty täydellisen polun kokonaiskustannus minkä tahansa siltasolmun kautta
    mu = float("inf")
    meeting_node = None

    def update_mu(node, total_cost):
        """
        Päivittää mu:n ja kohtaamissolmun, jos total_cost parantaa sitä.
        """
        nonlocal mu, meeting_node
        if total_cost < mu:
            mu = total_cost
            meeting_node = node

    # Vuorotellaan yksi laajennus kerrallaan (eteenpäin, taaksepäin, ...) ja tarkistetaan
    # pysäytysehto jokaisen yksittäisen laajennuksen jälkeen
    turn_forward = True
    while heap_f and heap_b:
        top_f = heap_f[0][0]
        top_b = heap_b[0][0]
        if top_f + top_b >= mu:
            break

        if turn_forward:
            expanded = _expand_one(
                heap_f, counter, g_f, came_from_f, closed_f, neighbor_fn, heuristic_fn, g_b, update_mu
            )
        else:
            expanded = _expand_one(
                heap_b, counter, g_b, came_from_b, closed_b, reverse_neighbor_fn, reverse_heuristic_fn, g_f, update_mu
            )

        if expanded is not None:
            nodes_expanded += 1

        turn_forward = not turn_forward

    if meeting_node is None or mu == float("inf"):
        # Yhteistä solmua ei löytynyt: polkua ei ole olemassa
        return BidirectionalResult(path=None, total_nost=None, nodes_expanded=nodes_expanded, meeting_node=None)
    
    # Rakennetaan polku alkupisteestä kohtaamispisteeseen (eteenpäin-taulukko),
    # ja sitten kohtaamispisteestä päätepisteeseen (taaksepäin-taulukko käänteisesti)
    forward_path = [meeting_node]
    cur = meeting_node
    while cur != start:
        cur = came_from_f[cur]
        forward_path.append(cur)
    forward_path.reverse()

    backward_path = []
    cur = meeting_node
    while cur != goal:
        cur = came_from_b[cur]
        backward_path.append(cur)
    
    full_path = forward_path + backward_path

    # Lasketaan lopullinen kustannus suoraan muodostetun polun kaarista sen sijaan, että
    # luotettaisiin väliaikaiseen mu-arvioon sellaisenaan. Tämä on tarpeen, koska
    # mu saatetaan päivittää hetkellä, jolloin jompikumpi suunta ei vielä ole sulkenut kohtaamissolmun g-arvoa
    actual_cost = 0.0
    for i in range(len(full_path) - 1):
        a, b = full_path[i], full_path[i + 1]
        edge_cost = None
        for neighbor, cost in neighbor_fn(a):
            if neighbor == b:
                edge_cost = cost
                break
        if edge_cost is None:
            raise RuntimeError(f"Kaarta {a} -> {b} ei löytynyt polun rakentamisessa.")
        actual_cost += edge_cost
    
    return BidirectionalResult(
        path=full_path, total_cost=actual_cost,
        nodes_expanded=nodes_expanded, meeting_node=meeting_node
    )