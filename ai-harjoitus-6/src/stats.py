"""
Dijkstran ja A*:n suorituskyvyn vertailu ja tilastointi.

Mittaa ja esittää empiirisesti väittämän: A* tutkii vähemmän solmuja kuin Dijkstra
löytäen silti optimaalisen polun kustannuksen, kunhan heuristiikka on ylärajaton ja kelvollinen.
"""

import time
from dataclasses import dataclass

from search_core import SearchResult

@dataclass
class RunStats:
    """
    Yhden algoritmiajon suoritustilastot:

    Attribuutit ja parametrit:
        name (str): Algoritmin/heuristiikan nimi
        result (SearchResult): Alkuperäinen SearchResult-olio
        elapsed_seconds (float): Ajon kesto sekunteina
    """
    name: str
    result: SearchResult
    elapsed_seconds: float

    @property
    def path_found(self):
        """
        Palauttaa True, jos polku löydettiin.
        """
        return self.result.path is not None
    
    @property
    def path_length(self):
        """
        Palauttaa polun solmujen määrän (0, jos polkua ei löytynyt).
        """
        return len(self.result.path) if self.result.path else 0
    
def time_run(name, search_callable):
    """
    Suorittaa hden hakufunktio ja mittaa sen suoritusajan.

    Parametrit:
        name (str): Ajon nimi
        search_callable (Any): Nollaparametrinen funktio (esim. lambda), joka suorittaa haun ja palauttaa SearchResult-olion

    Palauttaa:
        RunStats-olio, joka sisältää tuloksen ja ajan
    """
    t0 = time.perf_counter()
    result = search_callable()
    t1 = time.perf_counter()
    return RunStats(name=name, result=result, elapsed_seconds=t1 - t0)

def format_comparison_table(runs):
    """
    Muodostaa ihmisluettavan vertailutaulukon useasta ajosta.

    Parametrit:
        runs (list[RunStats]): Lista RunStats-oliota vertailtavaksi

    Palauttaa:
        Merkkijonomuotoinen taulukko, joka sisältää kunkin ajon nimen, löytyikö polku,
        polun kustannuksen, laajennettujen solmujen määrän, vertailtujen solmujen määrän ja suoritusajan
    """
    headers = ["Algoritmi", "Polku löytyi", "Kustannus", "Laajennetut solmut", "Vieraillut solmut", "Aika (ms)"]
    rows = []

    for run in runs:
        cost_str = f"{run.result.total_cost:.3f}" if run.result.total_cost is not None else "-"
        rows.append([
            run.name, "Kyllä" if run.path.found else "Ei",
            cost_str, str(run.result.nodes_expanded),
            str(run.result.nodes_visited), f"{run.elapsed_seconds * 1000:.3f}"
        ])

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def format_row(cells):
        return " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(cells))
    
    lines = [format_row(headers)]
    lines.append("-+-".join("-" * w for w in col_widths))
    for row in rows:
        lines.append(format_row(row))

    # Lisätään vertailuyheenveto, jos useampi kuin yksi ajo
    if len(runs) > 1:
        baseline = runs[0]
        lines.append("")
        for run in runs[1:]:
            if baseline.result.nodes_expanded > 0:
                ratio = run.result.nodes_expanded / baseline.result.nodes_expanded
                percentage = (1 - ratio) * 100
                if percentage >= 0:
                    comparison = (
                        f"{run.name} laajensi {run.result.nodes_expanded} solmua "
                        f"({percentage:.1f} % vähemmän kuin {baseline.name}, joka laajensi "
                        f"{baseline.result.nodes_expanded} solmua)"
                    )
                else:
                    comparison = (
                        f"{run.name} laajensi {run.result.nodes_expanded} solmua "
                        f"({-percentage:.1f} % vähemmän kuin {baseline.name}, joka laajensi "
                        f"{baseline.result.nodes_expanded} solmua)"
                    )
                lines.append(comparison)

    return "\n".join(lines)