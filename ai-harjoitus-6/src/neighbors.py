"""
Naapurisolujen generointi ruudukossa.

Tukee sekä 4-suuntaista että 8-suuntaista liikkumista. Diagonaalisiirtymän perushinta
on sqrt(2), koska se kuvaa Euklidista etäisyyttä yhden solmun diagonaalilla
(kun suora siirtymä on hinnaltaan 1).
"""

from math import sqrt

# 4-suuntaiset siirtymät: (dx, dy, kustannus)
ORTHOGONAL_MOVES = [
    (0, -1, 1.0), # pohjoinen
    (0, 1, 1.0), # etelä
    (1, 0, 1.0), # itä
    (-1, 0, 1.0) # länsi
]

# Diagonaaliset siirtymät: (dx, dy, kustannus)
DIAGONAL_MOVES = [
    (1, -1, sqrt(2)), # koillinen
    (1, 1, sqrt(2)), # kaakko
    (-1, 1, sqrt(2)), # lounas
    (-1, -1, sqrt(2)) # luode
]

def get_neighbors(grid, x, y, allow_diagonal = False, prevent_corner_cutting = True):
    """
    Palauttaa solmun (x,y) kävelykelpoiset naapurit ja siirtymäkustannukset.
    Kustannus lasketaan seuraavasti: siirtymän perushinta (suora = 1, diagonaali = sqrt(2)) kerrottuna kohdesolun maastopainolla.

    Parametrit:
        grid (grid.Grid): Ruudukko
        x (int): Nykyisen solun sarakeindeksi
        y (int): Nykyisen solun rivi-indeksi
        allow_diagonal (bool): Jos True, sallitaan myös 8-suuntainen liikkuminen
        prevent_corner_cutting (bool): Jos True (ja diagonaalit sallittu), diagonaalisiirtymä hylätään,
                                        mikäli jompikumpi viereisistä ortogonaalisista soluista on seinä

    Palauttaa:
        Lista pareja ((nx, ny), kustannus) jokaiselle kävelykelpiselle naapurille
    """
    neighbors = []

    moves = list(ORTHOGONAL_MOVES)
    if allow_diagonal:
        moves += DIAGONAL_MOVES

    for dx, dy, base_cost in moves:
        nx, ny = x + dx, y + dy

        if not grid.is_walkable(nx, ny):
            continue

        # Kulmien leikkaamisen esto: diagonaalisiirtymä vaatii, että
        # molemmat sivuavat ortogonaaliset solut ovat myös vapaita
        if allow_diagonal and prevent_corner_cutting and dx != 0 and dy != 0:
            side_a_walkable = grid.is_walkable(x + dx, y)
            side_b_walkable = grid.is_walkable(x, y + dy)
            if not (side_a_walkable and side_b_walkable):
                continue

        entry_cost = base_cost * grid.get_cost(nx, ny)
        neighbors.append(((nx, ny), entry_cost))

    return neighbors

def get_reverse_neighbors(grid, x, y, allow_diagonal = False, prevent_corner_cutting = True):
    """
    Palauttaa solut, joista solmuun (x,y) voidaan siirtyä, ja siirtymä kustannuksen ko. naapurista tähän soluun.

    Miksi? Siirtymän kustannus riippuu kohdesolun maastopainosta, ruudukko on itse asiassa suunnattu painotettu verkko, vaikka
    liikkuminen onkin geometrisesti symmetristä. Tämä tarkoittaa, että esim. kaksisuuntaisessa haussa taaksepäin suuntautuva haku ei voi käyttää samaa
    get_neighbors()-funktiota kuin eteenpäin haku sellaisennaan, vaan sen pitää käyttää tätä fukntiota, joka laskee kustannukset oikeasta suunnasta.

    Parametrit:
        grid (grid.Grid): Ruudukko, jossa naapuriteita etsitään
        x (int): Nykyisen solun sarakeindeksi
        y (int): Nykyisen solun rivi-indeksi
        allow_diagonal (bool): Jos True, sallitaan myös 8-suuntainen liikkuminen
        prevent_corner_cutting (bool): Ks. get_neighbors()

    Palauttaa:
        Lista pareja ((nx, ny), kustannus), missä kustannus on siirtymän hinta naapurista (nx, ny) tähän soluun
        (x,y) eli grid.get_cost(x, y) kerrottuna siirtymän perushinnalla.
    """
    neighbors = []

    if not grid.is_walkable(x, y):
        return neighbors
    
    entry_cost_to_here = grid.get_cost(x, y)

    moves = list(ORTHOGONAL_MOVES)
    if allow_diagonal:
        moves += DIAGONAL_MOVES

    for dx, dy, base_cost in moves:
        # Naapuri, josta voitaisiin siirtyä tähän soluun, on vastakkaiseen suuntaan
        px, py = x - dx, y - dy

        if not grid.is_walkable(px, py):
            continue

        if allow_diagonal and prevent_corner_cutting and dx != 0 and dy != 0:
            # Kulmien leikkaamisen esto lasketaan lähtösolun (px, py) näkökulmasta:
            # siirtymä px,py -> x,y vaatii, että molemmat sitä sivuavat ortogonaaliset solut (px+dx, py) ja (px, py+d), eli
            # (x, py) ja (px, y), ovat vapaita
            side_a_walkable = grid.is_walkable(x, py)
            side_b_walkable = grid.is_walkable(px, y)
            if not (side_a_walkable and side_b_walkable):
                continue

        cost_from_neighbor = base_cost * entry_cost_to_here
        neighbors.append(((px, py), cost_from_neighbor))

    return neighbors