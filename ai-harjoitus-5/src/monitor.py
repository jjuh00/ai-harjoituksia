"""
Telco-asiakaspysyvyynmallin seurantamoduuli.

Toteuttaa mallin suorituskyvyn seurannan, datan ajautumisen (drift) havaitsemisen ja
tilastollisten testien avulla tapahtuvan hälyttämisen.
"""

import os
import warnings
import pickle
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split

# Polut
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)

DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "WA_FN-UseC_-Telco-Customer-Churn.csv")
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
STATS_PATH = os.path.join(MODELS_DIR, "reference_stats.json")
REF_PREDS_PATH = os.path.join(MODELS_DIR, "reference_predictions.npy")
VAL_LABELS_PATH = os.path.join(MODELS_DIR, "validation_labels.npy")

# Kynnysarvot hälytyksille
PSI_WARNING_THRESHOLD = 0.1 # PSI >= 0.10 -> lievä muutos
PSI_ALERT_THRESHOLD = 0.20 # PSI >= 0.20 -> merkittävä muutos
KS_PVALUE_THRESHOLD = 0.05 # p-arvo < 0.05 -> tilastollisesti merkitsevä ajautuminen
PERFORMANCE_DROP_THRESHOLD = 0.05 # Suorituskyvyn lasku > 5% -> hälytys

# Ominaisuudet
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", 
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"
]
TARGET = "Churn"

class PSICalculator:
    """
    Laskee Population Stability Index (PSI) -arvon numeerisille ja kategorisille ominaisuuksille
    vertaamalla nykyistä dataa opetusdatan referenssitilastoihin.

    PSI on mittari, joka kuvaa kuinka paljon jakauma on muuttunut ajan myötä.
    PSI:n tulkinta:
        - PSI < 0.10: Ei merkittävää muutosta
        - 0.10 <= PSI < 0.20: Lievä muutos, seurattava
        - PSI >= 0.20: Merkittävä muutos, vaatii toimenpiteitä
    """

    def __init__(self, reference_stats, eps=1e-6):
        """
        Parametrit ja attribuutit:
            reference_stats (dict): Opetusdatan tilastot, jotka on laskettu ReferenceStatsBuilder-luokalla.
            eps (float): Pieni vakio, joka lisätään laskuihin nollalla jakamisen estämiseksi.
        """
        self.reference_stats = reference_stats
        self.eps = eps

    def calculate_numeric_psi(self, feature_name, current_series):
        """
        Laskee PSI-arvon numeeriselle ominaisuudelle käyttäen referenssihistorgrammin rajojen (bin edges) mukaisia pylväitä.

        Parametrit:
            feature_name (str): Ominaisuuden nimi, jonka PSI lasketaan.
            current_series (pd.Series): Nykyisen erän numeerinen sarja.

        Palauttaa:
            float: Laskettu PSI-arvo tai np.nan, jos refenssitiedot puuttuvat.
        """
        if feature_name not in self.reference_stats:
            warnings.warn(f"PSI-laskenta: Ominaisuutta {feature_name} ei löytynyt referenssitilastoista")
            return np.nan
        
        reference_info = self.reference_stats[feature_name]
        if reference_info.get("type") != "numeric":
            warnings.warn(f"PSI-laskenta: Ominaisuus {feature_name} ei ole numeerinen referenssitilastoissa")
            return np.nan
        
        current_clean = current_series.dropna()
        if current_clean.empty:
            warnings.warn(f"PSI-laskenta: Ominaisuudella {feature_name} ei ole nykyisessä erässä arvoja")
            return np.nan
        
        reference_proportions = np.array(reference_info["histogram"]["counts"])
        bin_edges = np.array(reference_info["histogram"]["bin_edges"])

        # Lasketaan nykyisen datan jakautuminen samoihin pylväisiin
        current_counts, _ = np.histogram(current_clean, bins=bin_edges)
        current_proportions = current_counts / (current_counts.sum() + self.eps)

        return self._psi_formula(reference_proportions, current_proportions)
    
    def calculate_categorical_psi(self, feature_name, current_series):
        """
        Laskee PSI-arvon kategoriselle ominaisuudelle vertaamalla nykyisten arvojen frekvenssejä referenssitilastoihin.

        Tuntemattomille kategorioille (joita ei ole referenssitilastoissa) käytetään eps-arvoa, jotta PSI-laskenta ei kaadu.

        Parametrit:
            feature_name (str): Ominaisuuden nimi, jonka PSI lasketaan.
            current_series (pd.Series): Nykyisen erän kategorinen sarja.

        Palauttaa:
            float: Laskettu PSI-arvo tai np.nan, jos referenssitiedot puuttuvat.
        """
        if feature_name not in self.reference_stats:
            warnings.warn(f"PSI-laskenta: Ominaisuutta {feature_name} ei löytynyt referenssitilastoista")
            return np.nan
        
        reference_info = self.reference_stats[feature_name]
        if reference_info.get("type") != "categorical":
            warnings.warn(f"PSI-laskenta: Ominaisuus {feature_name} ei ole kategorinen referenssitilastoissa")
            return np.nan
        
        current_clean = current_series.fillna("__missing__")
        if current_clean.empty:
            warnings.warn(f"PSI-laskenta: Ominaisuudella {feature_name} ei ole nykyisessä erässä arvoja")
            return np.nan
        
        reference_vc = reference_info["value_counts"]
        current_vc = current_clean.value_counts(normalize=True).to_dict()

        # Yhdistetään kaikki kategoriat molemmista jakaumista
        all_categories = set(reference_vc.keys()) | set(current_vc.keys())

        reference_proportions = np.array([reference_vc.get(cat, self.eps) for cat in all_categories])
        current_proportions = np.array([current_vc.get(cat, self.eps) for cat in all_categories])

        # Normalisoidaan varmistamaan, että summat ovat 1
        reference_proportions = reference_proportions / (reference_proportions.sum() + self.eps)
        current_proportions = current_proportions / (current_proportions.sum() + self.eps)

        return self._psi_formula(reference_proportions, current_proportions)
    
    def calculate_all_psi(self, current_df, numeric_features, categorical_features):
        """
        Laskee PSI-arvot kaikille numeerisille ja kategorisille ominaisuuksille.

        Parametrit:
            current_df (pd.DataFrame): Nykyinen erä DataFrame-muodossa.
            numeric_features (list[str]): Numeeriset ominaisuudet.
            categorical_features (list[str]): Kategoriset ominaisuudet.

        Palauttaa:
            dict: Sanakirja, jossa avaimina ominaisuuksien nimet ja arvoina PSI-arvot.
        """
        results = {}

        for feature in numeric_features:
            if feature in current_df.columns:
                results[feature] = self.calculate_numeric_psi(feature, current_df[feature])

        for feature in categorical_features:
            if feature in current_df.columns:
                results[feature] = self.calculate_categorical_psi(feature, current_df[feature])

        return results
    
    def _psi_formula(self, expected, actual):
        """
        Laskee PSI-arvon kaavalla PSI = sum((actual - expected) * ln(actual / expected)).

        Parametrit:
            expected (np.ndarray): Referenssijakauman osuudet.
            actual (np.ndarray): Nykyisen datan osuudet.

        Palauttaa:
            float: Laskettu PSI-arvo.
        """
        # Lisätään eps molempiin estämään log(0) ja nollalla jakaminen
        expected = np.where(expected == 0, self.eps, expected)
        actual = np.where(actual == 0, self.eps, actual)

        psi_value = np.sum((actual - expected) * np.log(actual / expected))
        return float(psi_value)
    
class KSTester:
    """
    Suorittaa Kolmogorov-Smirnov (KS) -testin numeerisille ominaisuuksille vertaamalla
    nykyistä dataa opetusdatan referenssijakaumaan.

    KS-testi mittaa kahden jakauman suurinta absoluuttista eroa. Pieni p-arvo (< 0.05) viittaa tilastollisesti
    merkittävään ajautumiseen.
    """

    def __init__(self, reference_stats, n_samples = 1000):
        """
        Parametrit ja attribuutit:
            reference_stats (dict): Opetusdatan tilastot, jotka on laskettu ReferenceStatsBuilder-luokalla.
            n_samples (int): Näytteiden määrä, joka otetaan referenssijakaumasta KS-testin suorittamiseksi (oletus 1 000).
        """
        self.reference_stats = reference_stats
        self.n_samples = n_samples
        self._rng = np.random.default_rng(seed=42)

    def perform_feature_ks_test(self, feature_name, current_series):
        """
        Suorittaa kaksisuuntaisen KS-testin yhdelle numeeriselle ominaisuudelle.

        Referenssijakaumaa lähestytään luomalla näytteitä histogrammipylväiden keskiluvuista painotettuna
        referenssijakauman osuuksilla.

        Parametrit:
            feature_name (str): Ominaisuuden nimi, jonka KS-testi suoritetaan.
            current_series (pd.Series): Nykyisen erän numeerinen sarja.

        Palauttaa:
            dict: Sanakirja, jossa avaimina:
                - "ks_statistic" (float): KS-tilastoarvo (0-1).
                - "p_value" (float): KS-testin p-arvo.
                - "drift_detected" (bool): True, jos p-arvo < 0.05, muuten False.
                - "error" (str | None): Virheilmoitus, jos testiä ei voitu suorittaa.
        """
        if feature_name not in self.reference_stats:
            return {"ks_statistic": np.nan, "p_values": np.nan,
                    "drift_detected": False, "error": f"Ominaisuutta {feature_name} ei löytynyt referenssitilastoista"}
        
        reference_info = self.reference_stats[feature_name]
        if reference_info.get("type") != "numeric":
            return {"ks_statistic": np.nan, "p_values": np.nan,
                    "drift_detected": False, "error": f"Ominaisuus {feature_name} ei ole numeerinen referenssitilastoissa"}
        
        current_clean = current_series.dropna().values
        if len(current_clean) < 2:
            return {"ks_statistic": np.nan, "p_values": np.nan,
                    "drift_detected": False, "error": f"Ominaisuudella {feature_name} ei ole riittävästi arvoja KS-testin suorittamiseksi"}
        
        # Luodaan referenssinäytteet histogrammijakaumasta
        refrence_samples = self._reconstruct_samples_from_histogram(reference_info["histogram"])

        ks_stat, p_value = stats.ks_2samp(refrence_samples, current_clean)

        return {
            "ks_statistic": float(ks_stat), "p_value": float(p_value),
            "drift_detected": bool(p_value < KS_PVALUE_THRESHOLD), "error": None
        }
    
    def perform_all_ks_test(self, current_df, numeric_features):
        """
        Suorittaa KS-testin kaikille numeerisille ominaisuuksille.

        Parametrit:
            current_df (pd.DataFrame): Nykyinen erä DataFrame-muodossa.
            numeric_features (list[str]): Numeeriset ominaisuudet.

        Palauttaa:
            dict: Sanakirja, jossa avaimina ominaisuuksien nimet ja arvoina KS-testin tulokset.
        """
        results = {}

        for feature in numeric_features:
            if feature in current_df.columns:
                results[feature] = self.perform_feature_ks_test(feature, current_df[feature])
        return results
    
    def _reconstruct_samples_from_histogram(self, histogram):
        """
        Luo likimääräisiä näytteitä histogrammidatasta käyttämällä pylväiden keskilukuja
        osuuksiensa mukaan painotettuina.

        Parametrit:
            histogram (dict): Sanakirja, joka sisältää listan pylväiden osuuksista ("counts") ja pylväiden rajoista ("bin_edges").

        Palauttaa:
            np.ndarray: Satunnaisia näytteitä muodostettuna histogrammijakaumasta.
        """
        proportions = np.array(histogram["counts"])
        edges = np.array(histogram["bin_edges"])
        bin_centers = (edges[:-1] + edges[1:]) / 2

        # Normalisoidaan painot
        weights = proportions / (proportions.sum() + 1e-10)

        chosen_bins = self._rng.choice(
            len(bin_centers), size=self.n_samples, p=weights
        )
        bin_widths = edges[1:] - edges[:-1]

        # Lisätään satunanista hajontaa luonnollisemman jakauman saamiseksi
        noise = self._rng.uniform(-0.5, 0.5, size=self.n_samples) * bin_widths[chosen_bins]
        samples = bin_centers[chosen_bins] + noise

        return samples
    
class PerformanceMonitor:
    """
    Laskee luokittelumittarit nykyiselle ennuste-erälle ja vertaa
    niitä referenssimittareihin suorituskyvyn ajautumisen havaitsemiseksi.
    """

    def __init__(self, reference_metrics=None):
        """
        Parametrit ja attribuutit:
            reference_metrics (dict | None): Sanakirja, joka sisältää referenssimittarit, 
            kuten roc_auc, tarkkuus, sisäinen tarkkuus, herkkyys ja F1-pisteet.
        """
        self.reference_metrics = reference_metrics

    def compute_metrics(self, y_true, y_pred, y_prob=None):
        """
        Laskee luokittelumittarit annetuille ennusteille.

        Parametrit:
            y_true (np.ndarray): Todelliset luokat.
            y_pred (np.ndarray): Ennustetut luokat.
            y_prob (np.ndarray | None): Ennustetut todennäköisyydet.

        Palauttaa:
            dict: Sanakirja, jossa avaimina mittarien nimet ja arvoina laskettu mittari.
        """
        return {
            "tarkkuus": float(accuracy_score(y_true, y_pred)),
            "f1_pisteet": float(f1_score(y_true, y_pred, zero_division=0)),
            "sisäinen_tarkkuus": float(precision_score(y_true, y_pred, zero_division=0)),
            "herkkyys": float(recall_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)) if y_prob is not None else np.nan
        }
    
    def compare_with_reference(self, current_metrics):
        """
        Vertaa nykyisiä mittareita referenssimittareihin ja tunnistaa merkittävät suorituskyvyn laskut.

        Parametrit:
            current_metrics (dict): Sanakirja, jossa avaimina mittarien nimet ja arvoina nykyiset mittarit.

        Palauttaa:
            dict: Vertailusanakirja, jossa jokaiselle mittarille:
                - "reference" (float): Referenssiarvo.
                - "current" (float): Nykyinen arvo.
                - "delta" (float): Nykyisen ja referenssin erotus.
                - "alert" (bool): True, jos suorituskyvyn lasku on merkittävä, muuten False.

            Palauttaa tyhjän sanakirjan, jos referenssimittareita ei ole asetettu.
        """
        if self.reference_metrics is None:
            return {}
        
        comparison = {}
        for metric, reference_value in self.reference_metrics.items():
            current_value = current_metrics.get(metric, np.nan)
            if isinstance(reference_value, (int, float)) and isinstance(current_value, (int, float)):
                delta = current_value - reference_value
                comparison[metric] = {
                    "reference": float(reference_value),
                    "current": float(current_value),
                    "delta": float(delta),
                    "alert": bool(delta < -PERFORMANCE_DROP_THRESHOLD)
                }
        return comparison
    
    def compute_reference_from_saved(self):
        """
        Lataa tallennetut ennusteet ja labelit MODELS_DIR:stä ja laskee niistä referenssimittarit.

        Palauttaa:
            dict | None: Referenssimittarisanakirja tai None, jos tiedostoja ei löydy.
        """
        try:
            y_pred = np.load(REF_PREDS_PATH)
            y_true = np.load(VAL_LABELS_PATH)
            # Todennäköisyyksiä ei ole tallennettu, joten ROC-AUC jätetään pois
            return self.compute_metrics(y_true, y_pred, y_prob=None)
        except FileNotFoundError:
            warnings.warn(
                "Referenssimittareita ei voitu laskea, koska tallennettuja ennusteita tai labeltiedostoja ei löytynyt. "
                "Suorituskykyvertailua ei siksi voida suorittaa"
            )
            return None
        
class MonitoringPipeline:
    """
    Pääseurantasilmukka, joka yhdistää PSI-laskennan, KS-testauksen ja suoritusykykyseurannan yhteen
    ajettavaan kokonaisuuteen.
    """

    def __init__(self):
        """
        Alustaa MonitoringPipelonen tyhjin aftefaktein.

        Attribuutit:
            model (sklearn.pipeline.Pipeline | None): Ladattu scikit-learn Pipeline tai None ennen lataamista.
            reference_stats (dict): Referenssitilastot tai tyhjä sanakirja.
            psi_calculator (PSICalculator | None): PSI-laskentakomponentti.
            ks_tester (KSTester | None): KS-testikomponentti.
            performance_monitor (PerformanceMonitor | None): Suorituskykyseurantakomponentti.
        """
        self.model = None
        self.reference_stats = {}
        self.psi_calculator = None
        self.ks_tester = None
        self.performance_monitor = None

    def load_artifacts(self):
        """
        Lataa kaikki seurantaan tarvittavat artefaktit levyltä:
            - Malli (model.pkl)
            - Referenssitilastot (reference_stats.json)
            - Referenssiennusteet (reference_predictions.npy)
        """
        # Ladataan malli
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Mallia ei löytynyt polusta {MODEL_PATH}. Suorita train.py ensin")
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        print(f"Malli ladattu ({MODEL_PATH})")

        # Ladataan referenssitilastot
        if not os.path.exists(STATS_PATH):
            raise FileNotFoundError(f"Referenssitilastoja ei löytynyt polusta {STATS_PATH}. Suorita train.py ensin")
        with open(STATS_PATH, 'r') as f:
            self.reference_stats = json.load(f)
        print(f"Referenssitilastot ladattu ({STATS_PATH})")

        # Alustetaan komponentit
        self.psi_calculator = PSICalculator(self.reference_stats)
        self.ks_tester = KSTester(self.reference_stats)

        # Suorituskykyseuranta: lasketaan referenssimittarit tallennetuista ennusteista
        _temp_monitor = PerformanceMonitor()
        reference_metrics = _temp_monitor.compute_reference_from_saved()
        self.performance_monitor = PerformanceMonitor(reference_metrics=reference_metrics)

        if reference_metrics:
            print("Referenssimittarit laskettu tallennetuista ennusteista")
        else:
            print("Referenssimittareita ei voitu laskea tallennetuista ennusteista. Suorituskykyvertailua ei voida suorittaa")

    def run(self, current_df, y_true=None):
        """
        Suorittaa seurantaprosessin kokonaisuudessaan annetulle dataerälle.

        Vaiheet:
            1. PSI-laskenta kaikille ominaisuuksille.
            2. KS-testi kaikille numeerisille ominaisuuksille.
            3. Malliennusteiden laskenta (jos malli on ladattu).
            4. Suorituskykymittarien laskenta (jos y_true on annettu).
            5. Suorituskyvyn vertailu referenssimittareihin.
            6. Hälytysten koostaminen.

        Parametrit:
            current_df (pd.DataFrame): Nykyinen dataerä, josta TARGET-sarake poistetaan automaattisesti, jos se on mukana.
            y_true (np.ndarray | None): Todelliset luokat nykyiselle erälle.

        Palauttaa:
            dict: Seurantasanakirja, joka sisältää:
                - "n_samples" (int): Nykyisen erän koko.
                - "psi" (dict): PSI-arvot ominaisuuksittain.
                - "ks_tests" (dict): KS-testin tulokset numeerisille ominaisuuksille.
                - "performance" (dict | None): Suorituskykymittarit tai None.
                - "performance_comparison" (dict | None): Vertailu referenssimittareihin tai None.
                - "alerts" (list[str]): Lista hälytyksistä, jotka on tunnistettu nykyisessä erässä.
        """
        if self.psi_calculator is None or self.ks_tester is None:
            raise RuntimeError("Artefakteja ei ole ladattu, kutsu ensin load_artifacts()-funktiota")
        
        # Poistetaan target-sarake, jos se on mukana
        X = current_df.drop(columns=[TARGET], errors="ignore").copy()
        n_samples = len(X)

        # Vaihe 1: PSI-laskenta
        print(f"Suoritetaan seurantaprosessi erälle, jossa on {n_samples} näytettä")
        print("1. Lasketaan PSI-arvot...")
        psi_results = self.psi_calculator.calculate_all_psi(X, NUMERIC_FEATURES, CATEGORICAL_FEATURES)

        # Vaihe 2: KS-testit numeerisille ominaisuuksille
        print("2. Suoritetaan KS-testit numeerisille ominaisuuksille...")
        ks_results = self.ks_tester.perform_all_ks_test(X, NUMERIC_FEATURES)

        # Vaihe 3: Malliennusteiden laskenta
        performace_metrics = None
        performance_comparison = {}

        print("3. Lasketaan malliennusteet...")
        if self.model is not None:
            try:
                y_prob = self.model.predict_proba(X)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)
            except Exception as e:
                warnings.warn(f"Ennusteiden laskenta epäonnistui: {e}")
                y_prob = None
                y_pred = None
        else:
            y_prob = None
            y_pred = None

        # Vaihe 4: Suorituskykymittarien laskenta
        print("4. Arvioidaan suorituskyky...")
        if y_true is not None and y_pred is not None:
            performace_metrics = self.performance_monitor.compute_metrics(y_true, y_pred, y_prob)
            performance_comparison = self.performance_monitor.compare_with_reference(performace_metrics)
        elif y_true is None:
            print("Todelliset luokat (y_true) eivät ole saatavilla, suorituskykymittareita ohitetaan")

        return {
            "n_samples": n_samples,
            "psi": psi_results,
            "ks_tests": ks_results,
            "performance": performace_metrics,
            "performance_comparison": performance_comparison,
            "alerts": self._collect_alerts(psi_results, ks_results, performance_comparison) # Vaihe 5: Hälytysten koostaminen
        }

    # Raportointi
    def print_report(self, report):
        """
        Tulostaa seurantaraportin luettavassa muodossa konsoliin.

        Parametrit:
            report (dict): Seurantasanakirja, joka on run()-funktion palauttama.
        """
        print("\n----- SEURANTARAPORTTI -----")
        print(f"Näytteiden määrä: {report['n_samples']}")
        # PSI-tulokset
        print("\nPSI-tulokset")
        print(f"{'Ominaisuus':<30} {'PSI':>8}  {'Tila'}")
        for feature, psi_value in sorted(report["psi"].items()):
            if np.isnan(psi_value):
                status = "N/A"
            elif psi_value >= PSI_ALERT_THRESHOLD:
                status = "HÄLTYTYS"
            elif psi_value >= PSI_WARNING_THRESHOLD:
                status = "VAROITUS"
            else:
                status = "OK"
            print(f"{feature:<30} {psi_value:>8.4f}  {status}")

        # KSI-testitulokset
        print("\nKS-testitulokset (numeeriset ominaisuudet)") 
        print(f"{'Ominaisuus':<20} {'KS-tilasto':>8}  {'p-arvo':>10}  {'Ajautuminen?'}")
        for feature, ks_result in sorted(report["ks_tests"].items()):
            if ks_result.get("error"):
                print(f"{feature:<20}  {ks_result['error']}")
                continue
            drift_str = "KYLLÄ" if ks_result["drift_detected"] else "EI"
            print(
                f"{feature:<20} {ks_result['ks_statistic']:>8.4f}  "
                f"  {ks_result['p_value']:>10.4f}\t{drift_str}"
            )

        # Suorituskykyvertailu
        if report["performance"]:
            print("\nSuorituskykymittarit")
            comparison = report["performance_comparison"]
            for metric, value in report["performance"].items():
                if np.isnan(value):
                    continue
                if metric in comparison:
                    c = comparison[metric]
                    alert_str = "HÄLTYTYS" if c["alert"] else "OK"
                    print(
                        f"{metric:<25} {value:.4f}  "
                        f"(referenssi: {c['reference']:.4f}, delta: {c['delta']:.4f})  {alert_str}"
                    )
                else:
                    print(f"{metric:<25} {value:.4f}  (referenssi: N/A)")

        # Hälytysten yhteenveto
        print(f"\nHälytysyhteenveto ({len(report['alerts'])} hälytystä)")
        if report["alerts"]:
            for alert in report["alerts"]:
                print(f"  [{alert['severity'].upper()}] {alert['message']}")

    def _collect_alerts(self, psi_results, ks_results, performace_comparison):
        """
        Koostaa hälytyslistan PSI-tulosten, KS-testitulosten ja suorituskykyvertailun perusteella.

        Parametrit:
            psi_results (dict): PSI-arvot
            ks_results (dict): KS-testin tulokset
            performace_comparison (dict): Suorituskykyvertailun tulokset

        Palauttaa:
            list[dict]: Lista hälityksistä, joissa jokainen hälytys on sanakirja:
                - "type" (str): Hältyksen tyyppi ("psi", "ks", "suorituskyky")
                - "feature" (str | None): Ominaisuuden nimi tai None.
                - "severity" (str): Hälytyksen vakavuus ("varoitus", "hälytys")
                - "message" (str): Kuvaus hälytyksestä.
                - "value" (float | None): PSI-arvo, p-arvo tai suorituskyvyn delta, joka aiheutti hälytyksen.
        """
        alerts = []

        # PSI-hälytykset
        for feature, psi_value in psi_results.items():
            if np.isnan(psi_value):
                continue
            if psi_value >= PSI_ALERT_THRESHOLD:
                alerts.append({
                    "type": "psi", "feature": feature, "severity": "hälytys",
                    "message": (
                        f"PSI-hälytys: {feature} on ajautunut merkittävästi (PSI={psi_value:.4f}), "
                        f"kun kynnys on {PSI_ALERT_THRESHOLD}"
                    ),
                    "value": psi_value
                })
            elif psi_value >= PSI_WARNING_THRESHOLD:
                alerts.append({
                    "type": "psi", "feature": feature, "severity": "varoitus",
                    "message": (
                        f"PSI-varoitus: {feature} on ajautunut lievästi (PSI={psi_value:.4f}), "
                        f"kun kynnys on {PSI_WARNING_THRESHOLD}"
                    ),
                    "value": psi_value
                })

        # KS-testihälytykset
        for feature, ks_result in ks_results.items():
            if ks_result.get("error"):
                continue
            if ks_result.get("drift_detected"):
                alerts.append({
                    "type": "ks", "feature": feature, "severity": "hälytys",
                    "message": (
                        f"KS-hälytys: {feature} on ajautunut merkittävästi "
                        f"(KS-tilasto={ks_result['ks_statistic']:.4f}, p-arvo={ks_result['p_value']:.4f})"
                    ),
                    "value": ks_result["ks_statistic"]
                })

        # Suorituskykyhälytykset
        for metric, comparison in performace_comparison.items():
            if comparison.get("alert"):
                alerts.append({
                    "type": "suorituskyky", "feature": None, "severity": "hälytys",
                    "message": (
                        f"Suorituskykyhälytys: {metric} on laskenut merkittävästi "
                        f"{comparison['delta']:+.4f} verrattuna referenssiin {comparison['reference']:.4f}, " 
                        f"kun nykyinen on {comparison['current']:.4f}"
                    ),
                    "value": comparison["delta"]
                })

        return alerts
    
def load_and_preprocess(path):
    """
    Lataa ja esikäsittelee Telco-asiakaspysyvyysdata DataFrame-muotoon. Peilaa train.py:n DataLoader.load()-funkiota.

    Parametrit:
        path (str): Polku CSV-tiedostoon, joka sisältää Telco-asiakaspysyvyysdatan.

    Palauttaa:
        pd.DataFrame: Esikäsitelty DataFrame, joka on valmis seurantaan.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Dataa ei löydy polusta {path}. Varmista, että data on ladattu ja polku on oikein")
        raise 

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.drop(columns=["customerID"], inplace=True)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    return df

# Pääohjelma
if __name__ == "__main__":
    print("Ladataan artefaktit...")
    pipeline = MonitoringPipeline()
    pipeline.load_artifacts()

    print("Ladataan ja esikäsitellään data seurantaan...")
    df = load_and_preprocess(DATA_DIR)

    # Otetaan viimeinen 20% datasta "uutena eränä" (samasta aineistosta, mutta eri näytteillä) ja käytetään sitä seurantaan
    _, current_batch = train_test_split(df, test_size=0.2, random_state=99, stratify=df[TARGET])

    y_true = current_batch[TARGET].values

    # Suoritetaan seurantaprosessi
    report = pipeline.run(current_batch, y_true=y_true)

    # Tulostetaan raportti
    pipeline.print_report(report)