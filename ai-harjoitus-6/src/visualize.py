"""
Ruudukon ja ratkaistun polun visualisointi ASCII-muodossa
Tarjoaa myös valinnaisen PNG-kuvan piirtämisen, jos matplotlib on asennettu.
"""

# Merkit polut ja tutkittujen solmujen esittämiseen
PATH_SYMBOL = "*"
VISITED_SYMBOL = "+"
START_SYMBOL = "A"
GOAL_SYMBOL = "L"

def render_grid(grid):
    """
    Muodostaa ruudukosta yksinkertaisen tekstiesityksen ilman polkua.

    Parametrit:
        grid (grid.Grid): Piirrettävä ruudukko

    Palauttaa:
        Merkkijono, jossa jokanen rivi vastaa yhtä ruudukon riviä
    """
    return "\n".join("".join(row) for row in grid.cells)

def render_path(grid, path, visited = None):
    """
    Muodostaa tekstiesityksen ruudukosta, jossa löydetty polku on
    merkitty '*'-merkillä ja tutkitut (mutta polkuun kuulumattomat) solut '+'-merkillä.

    Parametrit:
        grid (grid.Grid): Ruuduko, jolla haku suoritettiin
        path (list[tuple[int, int]] | None): Lista polun pisteistä (alusta loppuun), tai None jos polku ei löytynyt
        visited (set[tuple[int, int]] | None): Valinnainen joukko kaikista tutkituista soluista, havainnollistamaan
        kuinka laajalta algoritmi etsi

    Palauttaa:
        Merkkijono, jossa jokainen rivi vastaa yhtä ruudukon riviä, polku ja
        tutkitut solut merkittyinä.
    """
    # Kopioidaan solut, jotta alkuperäistä ruudukkoa ei muokata
    display = [row[:] for row in grid.cells]

    path_set = set(path) if path else set()

    if visited:
        for (vx, vy) in visited:
            if (vx, vy) in path_set:
                continue
            if display[vy][vx] in (".", ",", "^"):
                display[vy][vx] = VISITED_SYMBOL

    if path:
        for (px, py) in path:
            symbol = display[py][px]
            if symbol not in (START_SYMBOL, GOAL_SYMBOL):
                display[py][px] = PATH_SYMBOL

    return "\n".join("".join(row) for row in display)

def render_path_png(grid, path, output_path, visited = None):
    """
    Piirtää ruudukon ja löydetyn polun PNG-kuvaksi matplotlibin avulla.

    Tämä on valinnainen ominaisuus: jos matplotlib ei ole asennetut, funktio palauttaa
    False sen sijaan, että kaatuisi virheeseen.

    Parametrit:
        grid (grid.Grid): Ruudukko, jolla haku suoritettiin
        path (list[tuple[int, int]] | None): Lista polun pisteitä, tai None jos polkua ei löytynyt
        output_path (str): Tiedoston tallennuspolku
        visited (set[tuple[int, int]] | None): Valinnainen joukko tutkituista soluista

    Palauttaa:
        True, jos kuva piirrettiin ja tallennettiin onnistuneesti. False, jos matplotlib ei ole saatavilla
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        return False
    
    fig, axis = plt.subplots(figsize=(grid.width / 3, grid.height / 3))

    color_map = {
        "#": "#2b2b2b", # seinä
        ".": "#f5f5f5", # vapaa
        "^": "#304230", # raskas maasto
        ",": "#dfca99", # kevyt maasto
        "A": "#f5f5f5",
        "L": "#f5f5f6"
    }

    for y in range(grid.height):
        for x in range(grid.width):
            symbol = grid.cells[y][x]
            color = color_map.get(symbol, "#ffffff")
            axis.add_patch(patches.Rectangle((x, grid.height - 1 - y), 1, 1, facecolor=color, edgecolor="#cccccc", linewidth=0.3))

    if visited:
        for (vx, vy) in visited:
            if grid.cells[vy][vx] in ("#",):
                continue
            axis.add_patch(patches.Rectangle((vx, grid.height - 1 - vy), 1, 1, facecolor="#a9c9e8", alpha=0.5))

    if path:
        xk = [p[0] + 0.5 for p in path]
        yk = [grid.height - 1 - p[1] + 0.5 for p in path]
        axis.plot(xk, yk, color="#d1495b", linewidth=2.5, marker="o", markersize=3)

    if grid.start:
        ax, ay = grid.start
        axis.add_patch(patches.Rectangle((ax, grid.height - 1 - ay), 1, 1, facecolor="#1ccfba"))
    if grid.goal:
        lx, ly = grid.goal
        axis.add_patch(patches.Rectangle((lx, grid.height - 1 - ly), 1, 1, facecolor="#d95a3a"))

    axis.set_xlim(0, grid.width)
    axis.set_ylim(0, grid.height)
    axis.set_aspect("equal")
    axis.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return True