"""
Telco-asiakaspysyvyysmallin koulutus- ja arviointilogiikka.

Suoritusjärjestys:
    1. train.py: Kouluttaa mallin ja tallentaa sen levylle.
    2. monitor.py: Ajaa seurantaprosiessin ja tulostaa konsoliraportin.
    3. dashboard.py: Käynnistää Streamlit-sovelluksen, joka hakee seurantatiedot ja visualisoi ne.

Käyttö:
    python run_all.py # Koulutus + seuranta + Streamlit-sovellus
    python run_all.py --skip-train # Vain seuranta + Streamlit-sovellus (käytä olemassa olevaa mallia)
    python run_all.py --no-dashboard # Koulutus + seuranta, mutta ei Streamlit-sovellusta
"""

import os
import sys
import subprocess
import argparse
from sklearn.model_selection import train_test_split

# Polut
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(_CURRENT_DIR, "src")
TRAIN_SCRIPT = os.path.join(SRC_DIR, "train.py")
MONITOR_SCRIPT = os.path.join(SRC_DIR, "monitor.py")
DASHBOARD_SCRIPT = os.path.join(SRC_DIR, "dashboard.py")

def run_training():
    """
    Suorittaa koulutusvaiheen kutsumalla train.py:n train()-funktiota.

    Palauttaa:
        bool: True, jos koulutus onnistui, muuten False.
    """
    sys.path.insert(0, SRC_DIR)
    try:
        from src.train import train
        train()
        return True
    except Exception as e:
        print(f"Koulutus epäonnistui: {e}", file=sys.stderr)
        return False
    
def run_monitoring():
    """
    Suorittaa seurantaprosessin kutsumalla monitor.py:n logiikkaa suoraan.

    Palauttaa:
        bool: True, jos seuranta onnistui, muuten False.
    """
    sys.path.insert(0, SRC_DIR)
    try:
        from src.monitor import MonitoringPipeline, load_and_preprocess, DATA_DIR, TARGET
        pipeline = MonitoringPipeline()
        pipeline.load_artifacts()

        df = load_and_preprocess(DATA_DIR)
        _, current_batch = train_test_split(
            df, test_size=0.2, random_state=99, stratify=df[TARGET]
        )
        y_true = current_batch[TARGET].values

        report = pipeline.run(current_batch, y_true=y_true)
        pipeline.print_report(report)

        n_alerts = sum(1 for a in report["alerts"] if a["severity"] == "hälytys")
        print(f"\n Seuranta valmis, hälytyksiä: {n_alerts}")
        return True
    except Exception as e:
        print(f"Seuranta epäonnistui: {e}", file=sys.stderr)
        return False
    
def launch_dashboard():
    """
    Käynnistää Streamlit-sovelluksen subprocess.run-kutsulla.
    """
    print("\nKäynnistetään Streamlit-sovellus...")
    print("Avaa selaimessa: http://localhost:8501")
    print("Sulje sovellus painamalla Ctrl+C konsolissa\n")

    cmd = [sys.executable, "-m", "streamlit", "run", DASHBOARD_SCRIPT, "--server.headless", "true"]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nStreamlit-sovellus pysäytetty")
    except subprocess.CalledProcessError as e:
        print(f"Streamlit-sovellus kaatui koodilla {e.returncode}", file=sys.stderr)

def main():
    """
    Jäsentää komentoriviargumentit ja suorittaa putken vaiheet järjestyksessä:

    Argumentit (valinnaiset):
        --skip-train: Ohittaa koulutusvaiheen, käyttää olemassa olevaa mallia.
        --no-dashboard: Ohittaa Streamlit-sovelluksen käynnistyksen.
    """
    parser = argparse.ArgumentParser(description="Suorittaa koulutuksen, seurannan ja Streamlit-sovelluksen")
    parser.add_argument("--skip-train", action="store_true", help="Ohittaa koulutusvaiheen, käyttää olemassa olevaa mallia")
    parser.add_argument("--no-dashboard", action="store_true", help="Ohittaa Streamlit-sovelluksen käynnistyksen")
    args = parser.parse_args()

    print("Aloitetaan putki...\n")

    # Vaihe 1: Koulutus
    if not args.skip_train:
        ok = run_training()
        if not ok:
            print("Koulutus epäonnistui, putki keskeytetään virheen vuoksi", file=sys.stderr)
            sys.exit(1)
    else:
        print("Koulutus ohitettu, käytetään olemassa olevaa mallia\n")

    # Vaihe 2: Seuranta
    ok = run_monitoring()
    if not ok:
        print("Seuranta epäonnistui, putki keskeytetään virheen vuoksi", file=sys.stderr)
        sys.exit(1)

    # Vaihe 3: Streamlit-sovellus
    if not args.no_dashboard:
        launch_dashboard()
    else:
        print("Streamlit-sovellus ohitettu")

    print("\nPutki suoritettu onnistuneesti!")

if __name__ == "__main__":
    main()