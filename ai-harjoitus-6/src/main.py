"""
A* ja Dijkstra -reitinhakuohjelma.

Ohjelma tukee:
1. Ruudukkotilaa: lukee .txt-tiedostoista tai luo satunnaisen ruudukon, ja
   vertaa Dijkstraa ja A*:ia eri heuristiikoilla.
2. CSV-yleisverkkotilaa: lukee nodes.csv ja edges.csv -tiedostoa ja hkaee lyhyimmän
   polun Eujlidisella heuristiikalla.
3. Kaksisuuntaista hakua valinnaisena lisävertailuna.
"""

import argparse
import sys

from astar import astar
from bidirectional import bidirectional_search
from dijkstra import dijkstra
from graph import Graph
from grid import Grid
from heuristics import HEURISTICS
from neighbors import get_neighbors, get_reverse_neighbors
from stats import format_comparison_table, time_run
from visualize import render_path, render_path_png

def print_title(text):
    """
    Tulostaa otsikkorivin erottimella.

    Parametrit:
        text (str): Otsikon teksti
    """
    print("=" * len(text))
    print(text)
    print("=" * len(text))

def run_grid_mode(args):
    """
    Suorittaa ohjelman ruudukkotilassa: lataa tai generoi ruudukon, ajaa
    Dijkstran ja A*:n eri heuristiikoilla sekä tulostaa vertailun ja visualisoinnin.

    Parametrit:
        args (argparse.Namespace): Komentoriviparametrit

    Palauttaa:
        Paluuarvo kokonaislukuna (0 = onnistui, muu = virhe)
    """
    # Ruudukon lataus tai generointi
    if args.ruudukko:
        try:
            grid = Grid.from_file(args.ruudukko)
        except FileNotFoundError:
            print(f"Tiedostoa '{args.ruudukko}' ei löytynyt.")
            return 1
        except ValueError as e:
            print(f"Ruudukon jäsentely epäonnistui: {e}")
            return 1
    elif args.satunnainen:
        leveys, korkeus = args.satunnainen
        try:
            grid = Grid.generate_random_grid(
                width=leveys, height=korkeus,
                seed=args.siemenluku, wall_probability=args.seinatodennakoisyys
            )
        except ValueError as e:
            print(f"Satunnaisen ruudukon generointi epäonnistui: {e}")
            return 1
    else:
        print(f"Anna joko --ruudukko <tiedosto> tai --satunnaine <leveys> <korkeus>.")
        return 1
    
    # Tarkistetaan yleiset reunatapaukset
    if grid.start is None:
        print(f"Ruudukosta ei löytynyt lähtöpistettä A.")
        return 1
    if grid.goal is None:
        print(f"Ruudukosta ei löytynyt päätepistellä L.")
        return 1
    if grid.start == grid.goal:
        print("Alku- ja päätepiste ovat samat, polun pituus on 0.")

    
    print_title(f"Ruudukko ladattu ({grid.width}x{grid.height})")
    print(f"Alku: {grid.start}  Pääte: {grid.goal}")
    print(f"Liikkumismalli: {'8-suuntainen (diagonaalit sallittu)' if args.diagonaali else '4-suuntainen'}")

    if args.nayta_ruudukko:
        print()
        from visualize import render_grid
        print(render_grid(grid))

    # Ajetaan Dijkstra
    runs = []
    runs.append(
        time_run("Dijkstra", lambda: dijkstra(grid, grid.start, grid.goal, allow_diagonal=args.diagonaali))
    )

    # Ajetaan A* valituilal heuristiikoilla
    heuristic_names = args.heuristiikat if args.heuristiikat else (["octile"] if args.diagonaali else ["manhattan"])
    fi_to_func = {
        "manhattan": HEURISTICS["manhattan"],
        "euklidinen": HEURISTICS["euklidinen"],
        "chebyshev": HEURISTICS["chebyshev"],
        "octile": HEURISTICS["octile"]
    }

    for name in heuristic_names:
        if name not in fi_to_func:
            print(f"Tuntematon heuristiikka '{name}', ohitetaan. Sallitut: {list(fi_to_func.keys())}")
            continue
        heuristic_fn = fi_to_func[name]
        runs.append(
            time_run(
                f"A* ({name})",
                lambda hf=heuristic_fn: astar(grid, grid.start, grid.goal, hf, allow_diagonal=args.diagonaali)
            )
        )

    # Käsitellään reunatapaus, jossa polkua ei ole olemassa
    if runs[0].result.path is None:
        print()
        print("Polkua alkupisteestä päätepisteeseen EI löytynyt (päätepiste ei ole saavutettavissa).")
        print()
        print(format_comparison_table(runs))
        return 0
    
    print_title("Tulokset")
    print(format_comparison_table(runs))

    # Kaksisuuntainen haku
    if args.kaksisuuntainen:
        def nfn(node):
            x, y = node
            return get_neighbors(grid, x, y, allow_diagonal=args.diagonaali)
        
        def rnfn(node):
            x, y = node
            return get_reverse_neighbors(grid, x, y, allow_diagonal=args.diagonaali)
        
        import time as _time
        t0 = _time.perf_counter()
        bidir_result = bidirectional_search(grid.start, grid.goal, nfn, None, rnfn, None)
        t1 = _time.perf_counter()

        print()
        print("Kaksisuuntainen Dijkstra:")
        print(f"Kustannus: {bidir_result.total_cost}")
        print(f"Laajennetut solmut (molemmat suunnat yhteensä): {bidir_result.nodes_expanded}")
        print(f"Aika: {(t1 - t0) * 1000:.3f} ms")
        print(f"Kohtaamissolmu: {bidir_result.meeting_node}")

    # Visualisointi
    best_run = runs[-1] if len(runs) > 1 else runs[0]
    print_title(f"Polku ({best_run.name})")
    print(render_path(grid, best_run.result.path, visited=set(best_run.result.g_score.keys())))

    if args.tallenna_kuva:
        ok = render_path_png(grid, best_run.result.path, args.tallenna_kuva, visited=set(best_run.result.g_score.keys()))
        if ok:
            print(f"\nKuva tallennettu: {args.tallenna_kuva}")
        else:
            print(f"matplotlib ei ole asennettu, PNG-kuvaa ei voitu tallentaa.")

    return 0

def run_graph_mode(args):
    """
    Suorittaa ohjelman yleisen CSV-verkon tilassa.

    Parametrit:
        args (argparse.Namespace): Komentoriviparametrit

    Palauttaa:
        Paluuarvo kokonaislukuna (0 = onnistui, muu = virhe)
    """
    try:
        graph = Graph.from_csv(args.solmut, args.kaaret)
    except FileNotFoundError:
        print(f"Tiedostoa ei löytynyt.")
        return 1
    except ValueError as e:
        print(f"CSV-tiedoston jäsentely epäonnistui: {e}")
        return 1
    
    if args.lahto not in graph.coords:
        print(f"Alkusolmua '{args.lahto}' ei löytynyt nodes.csv-tiedostosta.")
        return 1
    if args.maali not in graph.coords:
        print(f"Päätesolmua '{args.maali}' ei löytynyt nodes.csv-tiedostosta.")
        return 1
    
    print_title(f"Verkko ladattu {len(graph.coords)} solmulla.")
    print(f"Alku: {args.lahto}  Pääte: {args.maali}")

    from search_core import best_first_search

    runs = []
    runs.append(
        time_run("Dijkstra (verkko)", lambda: best_first_search(args.lahto, args.maali, graph.neighbors, heuristic_fn=None))
    )
    runs.append(
        time_run("A* (verkko, euklidinen)",
            lambda: best_first_search(
                args.lahto, args.maali, graph.neighbors, heuristic_fn=lambda n: graph.heuristic(n, args.maali)
            )
        )
    )

    if runs[0].result.path is None:
        print("\nPolkua ei löytynyt.")
        print()
        print(format_comparison_table(runs))
        return 0
    
    print_title("Tulokset")
    print(format_comparison_table(runs))

    print()
    print("Löydetty polku:", " -> ".join(str(n) for n in runs[-1].result.path))

    return 0

def build_parser():
    """
    Rakentaa komentoriviparametrien jäsentimen.

    Palauttaa:
        Määriteltu ArgumentParser-olio
    """
    parser = argparse.ArgumentParser(
        description="A* ja Dijkstra -reitinhakuohjelma. Vertailee kahta hakualgoritmia samalla ruudukolla tai "
        "verkolla, ja näyttää tilastot sekä ratkaistun polun.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    group_source = parser.add_mutually_exclusive_group()
    group_source.add_argument("--ruudukko", type=str, help="Polku ruudukkotidostoon (.txt).")
    group_source.add_argument("--satunnainen", type=int, nargs=2, metavar=("LEVEYS", "KORKEUS"), help="Generoi satunnainen ruudukko.")

    parser.add_argument("--siemenluku", type=int, default=None, help="Satunnaislukugeneraattorin siemenluku.")
    parser.add_argument("--seinatodennakoisyys", type=float, default=0.25, help="Seinän todennäköisyys satunnaisessa ruudukossa.")
    parser.add_argument("--diagonaali", action="store_true", help="Salli 8-suuntainen liikkuminen.")
    parser.add_argument(
        "--heuristiikat",
        nargs="+",
        choices=["manhattan", "euklidinen", "chebyshev", "octile"],
        help="Mitä heuristiikkoja A*:lle testataan (oletus octile jos diagonaali, muuten manhattan)."
    )
    parser.add_argument("--kaksisuuntainen", action="store_true", help="Aja kaksisuuntainen Dijkstra vertailuna.")
    parser.add_argument("--nayta-ruudukko", action="store_true", help="Tulosta alkuperäinen ruudukok ennen hakua.")
    parser.add_argument("--tallenna-kuva", type=str, default=None, help="Tallenna PNG-kuva annettuun polkuun (vaatii matplotlib).")
    parser.add_argument("--solmut", type=str, help="Polku nodes.csv-tiedostoon (CSV-verkkotilassa).")
    parser.add_argument("--kaaret", type=str, help="Polku edges.csv-tiedostoon (CSV-verkkotilassa).")
    parser.add_argument("--lahto", type=str, help="Alkusolmun tunniste (CSV-verkkotilassa).")
    parser.add_argument("--maali", type=str, help="Päätesolmun tunniste (CSV-verkkotilassa).")

    return parser

def main():
    """
    Ohjelman pääfunktio, joka jäsentelee komentoriviparametrit ja
    käyninstää joko ruudukko- tai verkkotilan.

    Palauttaa:
        Paluuarvo kokonaislukuna (0 = onnistui, muu = virhe)
    """
    parser = build_parser()
    args = parser.parse_args()

    # CSV-verkkotila tunnistetaan siitä, että --solmut ja --kaaret on annettu
    if args.solmut or args.kaaret:
        if not (args.solmut and args.kaaret and args.lahto and args.maali):
            print("CSV-verkkotila vaatii kaikki parametrit: --solmut, --kaaret, --lahto, --maali")
            return 1
        return run_graph_mode(args)
    
    if not args.ruudukko and not args.satunnainen:
        print("Anna joko --ruudukko, --satunnainen tai CSV-verkkotilan parametrit (--solmut/--kaaret).")
        print("Käytä --help nähdäksesi kaikki parametrit.")
        return 1
    
    return run_grid_mode(args)

if __name__ == "__main__":
    sys.exit(main())