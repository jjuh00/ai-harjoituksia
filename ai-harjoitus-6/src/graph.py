"""
Yleinen painotettu verkko CSV-tiedostoista ruudukon sijaan. Tämä yleistää A*:n ruudukon ulkopuolelle samaan muotoon, 
jota käytetään oikeissa tieverkkoreitityssovelluksissa: solmuilla on koordinaatit ja kaarilla on painot, jotka eivät välttämättä ole suoraan
koordinaattien välinen etäisyys.

Kaaret tulkitaan suuntaamattomiksi: jos A->B on kaari, myös B->A on kulkukelpoinen samalla painolla, ellei toisin mainita.

Heuristiikkana käytetään solmujen koordinaattien välistä Euklidista etäisyyttä. Tämä on kelvollinen, jos kaarien painot
ovat vähintään yhtä suuria kuin solmujen väline Euklidinen etäisyys.
"""

import csv
from dataclasses import dataclass, field
from math import sqrt

@dataclass
class Graph:
    """
    Yleinnen painotettu, suuntaamaton verkko koordinaateilla varustetuulle solmuille.

    Attribuutit ja parametrit:
        coords (dict[str, tuple[float, float]]): Sanakirja node_id -> (x,y)
        adjacency (dict[str, list[tuple[str, float]]]): Sanakirja node_id -> lista (neighbor_id, weight) -pareja
    """
    coords: dict[str, tuple[float, float]] = field(default_factory=dict)
    adjacency: dict[str, list[str, float]] = field(default_factory=dict)

    def neighbors(self, node):
        """
        Palauttaa solmun naapurit ja kaarien painot.

        Parametrit:
            node (str): Solmun tunniste

        Palauttaa:
            Lista (neighbor_id, paino) -pareja. Tyhjä lista, jos solmulla ei ole kaaria
        """
        return self.adjacency.get(node, [])
    
    def heuristic(self, a, b):
        """
        Euklidinen etäisyys kahden solmun koordinaattien välillä; toimii A*:n
        heuristiikkana yleiselel verkolle.

        Parametrit:
            a (str): Ensimmäisen solmun tunniste
            b (str): Toisen solmun tunniste

        Palauttaa:
            Solmujen koordinaattien välinen Euklidinen etäisyys

        Nostaa:
            KeyError, jos jommankumman solmun koordinaatteja ei tunneta
        """
        ax, ay = self.coords[a]
        bx, by = self.coords[b]
        return sqrt((ax - bx) ** 2 + (ay - by) ** 2)
    
    @classmethod
    def from_csv(cls, nodes_path, edges_path):
        """
        Lukee verkon kahdesta CSV-tiedosta.

        Parametrit:
            nodes_path (str): Polku solmut sisältävään CSV-tiedostoon
            edges_path (str): Polku kaaret sisältävään CSV-tiedostoon

        Palauttaa: Graph-olio

        Nostaa:
            ValueError, jos tiedostoissa on virheellisiä rivejä
            FileNotFoundError, jos jompikumpi tiedosto puuttuu
        """
        coords = {}

        with open(nodes_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            required = {"node_id", "x", "y"}
            if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
                raise ValueError(
                    f"nodes.csv-tiedoston sarakkeiden on oltava {sorted(required)}, löytyi {reader.fieldnames}."
                )
            for i, row in enumerate(reader):
                try:
                    node_id = row["node_id"].strip()
                    x = float(row["x"])
                    y = float(row["y"])
                except (ValueError, AttributeError) as e:
                    raise ValueError(f"Virheellinen rivi {i + 1} tiedostossa {nodes_path}: {row}") from e
                
                if not node_id:
                    raise ValueError(f"Tyhjä solmun tunniste rivillä {i + 2} tiedostossa {nodes_path}.")
                coords[node_id] = (x, y)

        adjacency = {node_id: [] for node_id in coords}

        with open(edges_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            required = {"start_node", "target_node", "weight"}
            if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
                raise ValueError(
                    f"edges.csv-tiedoston sarakkeiden on oltava {sorted(required)}, löytyi {reader.fieldnames}."
                )
            for i, row in enumerate(reader):
                try:
                    src = row["start_node"].strip()
                    dst = row["target_node"].strip()
                    weight = float(row["weight"])
                except (ValueError, AttributeError) as e:
                    raise ValueError(f"Virheellinen rivi {i + 2} tiedostossa {edges_path}: {row}") from e
                
                if src not in coords:
                    raise ValueError(f"Kaari viittaa tuntemattomaan solmuun '{src}' rivillä {i + 2}.")
                if dst not in coords:
                    raise ValueError(f"Kaari viittaa tuntemattomaan solmuun '{dst}' rivillä {i + 2}.")
                if weight < 0:
                    raise ValueError(f"Negatiivinen paino ei ole sallittu (rivi {i + 2}).")
                
                adjacency[src].append((dst, weight))
                adjacency[dst].append((src, weight))

        return cls(coords=coords, adjacency=adjacency)