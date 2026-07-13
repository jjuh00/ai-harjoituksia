"""
Ruudukon esitys ja jäsentely.
Ruudukko luetaan tekstitiedostosta, jossa jokainen rivi vastaa yhtä ruudukon riviä
ja jokainen merkki yhtä solua: Tuetut merkit:
    '.': vapaa solu (kävelykelpoinen, paino 1.0)
    '#': este (seinä, ei kävelykelpoinen)
    'A': lähtöpoiste
    'L': päätepiste
    '^': raskas maasto (esim. suo), paino 3.0
    ',': kevyt maasto (esim. hiekka), paino 1.5

Ruudukko on suorakulmainen eli kaikkien rivien on oltava yhtä pitkiä.
"""

from dataclasses import dataclass, field

# Solutyyppien painot
TERRAIN_WEIGHTS = {
    ".": 1.0,
    "A": 1.0,
    "L": 1.0,
    "^": 3.0,
    ",": 1.5
}

WALL = "#"

@dataclass
class Grid:
    """
    Yksinkertainen 2D-ruudukko, jota käytetään reitinhaun perusympäristönä.

    Attribuutit ja parametrit:
        width (int): Ruudukon leveys (sarakkeiden lkm)
        height (int): Ruudukon korkeus (rivien lkm)
        cells (list[list[str]]): Kaksiulotteinen lista merkkejä
        start (tuple[int, int] | None): Lähtösolun koordinaatit (x,y)
        goal (tuple[int, int] | None): Päätesolun koordinaatit (x,y)
    """
    width: int
    height: int
    cells = field(default_factory=list)
    start = None
    goal = None

    def in_bounds(self, x, y):
        """
        Tarkistaa, onko koordinaatti ruudukon sisällä.

        Parametrit:
            x (int): Sarakeindeksi
            y (int): Rivi-indeksi

        Palauttaa:
            True, jos (x,y) on ruudukon rajojen sisällä, muuten False
        """
        return 0 <= x < self.width and 0 <= y < self.height
    
    def is_walkable(self, x, y):
        """
        Tarkistaa, voiko solun (x,y) läpi kulkua.

        Parametrit:
            x (int): Sarakeindeksi
            y (int): Rivi-indeksi

        Palauttaa:
            True, jos solu on ruudukon sisällä eikä ole seinä ('#')
        """
        if not self.in_bounds(x, y):
            return False
        return self.cells[y][x] != WALL
    
    def get_cost(self, x, y):
        """
        Laskee solun (x,y) sisääntulokustannuksen eli maaston painon.

        Parametrit:
            x (int): Sarakeindeksi
            y (int): Rivi-indeksi

        Palauttaa:
            Solun paino TERRAIN_WEIGHTS-taulukosta. Oletusarvo 1.0, jos
            merkki on tuntematon mutta kävelykelpoinen

        Nostaa:
            ValueError, jos solu ei ole kävelykelpoinen
        """
        if not self.is_walkable(x, y):
            raise ValueError(f"Solu ({x},{y}) ei ole kävelykelpoinen.")
        symbol = self.cells[y][x]
        return TERRAIN_WEIGHTS.get(symbol, 1.0)
    
    @classmethod
    def from_text(cls, text):
        """
        Jäsentää ruudukon monirivisestä merkkijonosta.

        Params:
            text (str): Ruudukon tekstiesitys, rivit erotettu '\\n'-merkillä.
                        Tyhjät rivit tekstin alusta ja lopusta ohitetaan

        Palauttaa:
            Uusi Grid-olio, jonka start/goal on asetettu, jos merkit 'A' ja/tai 'L' löytyivät.

        Nostaa:
            ValueError, jos rivit eivät ole yhtä pitkiä tai ruudukko on tyhjä,
                        tai jos 'A'/'L' esiintyy useammin kuin kerran.
        """
        raw_lines = text.splitlines()
        # Poistetaan tyhjät rivit
        lines = [line.strip("\r") for line in raw_lines if line.rstrip("\r") != ""]

        if not lines:
            raise ValueError("Ruudukko on tyhjä, tiedostoa ei löytynyt rivejä.")
        
        width = len(lines[0])
        for i, line in enumerate(lines):
            if len(line) != width:
                raise ValueError(f"Rivi {i} on eri pituinen {len(line)} kuin ensimmäinen rivi ({width}). Ruudukon on oltava suorakulmainen.")
            
        height = len(lines)
        cells = [list(line) for line in lines]

        start = None
        goal = None

        for y in range(height):
            for x in range(width):
                symbol = cells[y][x]
                if symbol == "A":
                    if start is not None:
                        raise ValueError("Useampi kuin yksi lähtöpiste 'A' löytyi.")
                    start = (x, y)
                elif symbol == "L":
                    if goal is not None:
                        raise ValueError("Useampi kuin yksi päätepiste 'L' löytyi.")
                    goal = (x, y)

        return cls(width=width, height=height, cells=cells, start=start, goal=goal)
    
    @classmethod
    def from_file(cls, path):
        """
        Lukee ruudukon tiedostosta ja jäsentää sen.

        Parametrit:
            path (str): Polku tekstitiedostoon

        Palauttaa:
            Uusi Grid-olio

        Nostaa:
            FileNotFoundError, jos tiedostoa ei löytynyt.
            ValueError, jos tiedoston sisältö on virheellinen (ks. from_text)
        """
        with open(path, 'r', encoding="utf-8") as f:
            text = f.read()
        return cls.from_text(text)
    
    @classmethod
    def generate_random_grid(
        cls, width, height, seed = None,
        wall_probability = 0.25, heavy_probability = 0.05, light_probability = 0.10
    ):
        """
        Luo satunnaisen ruudukon demoja varten.

        Lähtöpiste asetetaan aina vasempaan yläkulmaan (0,0) ja
        päätepiste oikeaan alakulmaan (width-1,height-1).

        Parametrit:
            width (int): Ruudukon leveys
            height (int): Ruudukon korkeus
            seed (int | None): RNG-siemenluku
            wall_probability (float): Todennäköisyys, että solu on seinä
            heavy_probability (float): Todennäköisyys raskaalle maastolle ('^')
            light_probability (float): Todennäköisyys kevyelle maastolle (',')

        Palauttaa:
            Uusi satunnaisesti generoitu Grid-olio

        Nostaa:
            Valuerror, jos width tai height on pienempi kuin 2
        """
        import random as rnd

        if width < 2 or height < 2:
            raise ValueError("Ruudukon leveyden ja korkeudene on oltava vähintään 2.")
        
        rng = rnd.Random(seed)
        cells = []
        for _y in range(height):
            row = []
            for _x in range(width):
                r = rng.random()
                if r < wall_probability:
                    row.append(WALL)
                elif r < wall_probability + heavy_probability:
                    row.append("^")
                elif r < wall_probability + heavy_probability + light_probability:
                    row.append(",")
                else:
                    row.append(".")
            cells.append(row)
        
        cells[0][0] = "A"
        cells[height - 1][width - 1] = "L"

        return cls(width=width, height=height, cells=cells, start=(0, 0), goal=(width - 1, height - 1))