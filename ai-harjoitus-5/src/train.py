"""
Televiestintäalan (Telco) asiakaspysyvyysmallin koulutusmalli.
"""

import os
import pickle
import json
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split

# Polut data- ja mallihakemistoihin
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)

DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "WA_FN-UseC_-Telco-Customer-Churn.csv")
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")

class DataLoader:
    """Lataa ja siistii raakadatan."""

    def __init__(self, path):
        """
        Parametrit ja attribuutit:
            path (str): Polku CSV-tiedostoon, joka sisältää raakadatan.
        """
        self.path = path

    def load(self):
        """
        Lukee CSV-tiedot, poistaa ID-sarakkeen ja enkoodaa binäärimuuttujan.
        
        Palauttaa:
            pd.DataFrame: Siistitty dataframe valmiina mallinnukseen.
        """
        try:
            df = pd.read_csv(self.path)
        except FileNotFoundError:
            print(f"Dataa ei löydy polusta {self.path}. Varmista, että data on ladattu ja polku on oikein")
            raise

        # Muutetaan TotalCharges numeeriseksi
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        # Poistetaan ID-sarake, joka ei ole hyödyllinen mallinnuksessa
        df.drop(columns=["customerID"], inplace=True)

        # Binäärimuuttujan enkoodaus: "Yes" -> 1, "No" -> 0
        df["Churn"] = (df["Churn"] == "Yes").astype(int)

        return df
    
# Ominaisuudet
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"
]
TARGET = "Churn"

class ReferenceStatsBuilder:
    """
    Laskee ominaisuuskohtaiset tilastot opetusdatasta.
    
    Näitä tilastoja käytetään myöhemmin poikkeamien tunnistamiseen opetusdatan ja tulevien erien välillä.
    """

    def __init__(self, numeric_features, categorical_features):
        """
        Parametrit ja attribuutit:
            numeric_features (list[str]): Lista numeerisista ominaisuuksista.
            categorical_features (list[str]): Lista kategorisista ominaisuuksista.
        """
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

    def build(self, df):
        """
        Laskee tilastot jokaiselle ominaisuudelle.

        Numeerisille omiaisuuksille sanakirja sisältää: keskiarvo, keskihajonta, mini, maksimi, mediaani, kvantiilit (25%, 75%), puuttuvien arvojen aste.

        Kategorisille ominaisuuksille sanakirja sisältää: frekvenssit ja puuttuvien arvojen aste.

        Parametrit:
            df (pd.DataFrame): Opetusdata dataframe-muodossa.

        Palauttaa:
            dict: Sanakirja, jossa avaimina ominaisuuksien nimet, uniikit (lkm) ja arvoina tilastotiedot.
        """
        stats = {}

        for column in self.numeric_features:
            series = df[column].dropna()
            stats[column] = {
                "type": "numeric",
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "median": float(series.median()),
                "q25": float(series.quantile(0.25)),
                "q75": float(series.quantile(0.75)),
                "missing_rate": float(df[column].isna().mean()),
                # Tallennetaan histogrammitiedot poikkeamien tunnistusta varten
                "histogram": self._histogram(series)
            }
        
        for column in self.categorical_features:
            series = df[column].fillna("__missing__")
            vc = series.value_counts(normalize=True).to_dict()
            stats[column] = {
                "type": "categorical",
                "value_counts": {str(key): float(value) for key, value in vc.items()},
                "n_unique": int(df[column].nunique()),
                "missing_rate": float(df[column].isna().mean())
            }

        return stats

    def _histogram(self, series, bins=10):
        """
        Laskee PSI-laskentaa varten sopivan histogrammin, jossa on kiinteä määrä pylväitä.

        Parametrit:
            series (pd.Series): Numeerinen sarja.
            bins (int): Pylväiden määrä histogrammissa (oletus 10).

        Palauttaa:
            dict: Sanakirja, jossa osuudet (counts) on lista pylväiden osuuksista ja rajat (bin_edges) on lista pylväiden rajapisteistä.
        """
        counts, edges= np.histogram(series, bins=bins)
        # Normalisoidaan osuudet siten, että PSI-laskenta toimii oikein
        proportions = (counts / counts.sum()).tolist()
        return {
            "counts": proportions,
            "bin_edges": edges.tolist()
        }
    
def build_pipeline():
    """
    Muodostaa kokonaisen scikit-learn mallinnusputken (esikäsittely + luokitin).

    Esikäsittely
    - Numeeriset ominaisuudet: Imputointi mediaanilla, skaalaus StandardScalerillä.
    - Kategoriset ominaisuudet: Vakioarvojen imputointi, one-hot koodaus.

    Luokitin:
    Käytetään GradientBoostingClassifieria järkevillä oletusarvoilla taulukkomuotoiseen binääriluokitteluun.

    Palauttaa:
        sklearn.pipeline.Pipeline: Koko mallinnusputki, joka voidaan sovittaa opetusdataan ja käyttää ennustamiseen.
    """
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, NUMERIC_FEATURES), ("cat", categorical_transformer, CATEGORICAL_FEATURES)]
    )

    clf = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=42
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])

def evaluate(y_true, y_pred, y_prob):
    """
    Laskee luokittelumittarit testidatasta.

    Parametrit:
        y_true (np.ndarray): Todelliset luokat.
        y_pred (np.ndarray): Ennustetut luokat.
        y_prob (np.ndarray): Ennustetut todennäköisyydet.

    Palauttaa:
        dict: Sanakirja, jossa avaimina mittarien nimet ja arvoina laskettu mittari.
    """
    return {
        "tarkkuus": accuracy_score(y_true, y_pred),
        "f1_pisteet": f1_score(y_true, y_pred),
        "sisäinen_tarkkuus": precision_score(y_true, y_pred),
        "herkkyys": recall_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }

def train():
    """Päästä päähän koulutusprosessi; tallentaa aftefaktit models-kansioon."""
    # Ladataan data
    print("Ladataan data...")
    loader = DataLoader(DATA_DIR)
    df = loader.load()

    X = df.drop(columns=[TARGET])
    y = df[TARGET].values

    # Jaetaan data opetus- ja testiaineistoon
    print("Jaetaan data opetus- ja testiaineistoon (80 / 20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Lasketaan tilastot opetusdatasta
    print("Lasketaan tilastot opetusdatasta...")
    stats_builder = ReferenceStatsBuilder(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    ref_stats = stats_builder.build(X_train)

    # Koulutetaan malli
    print("Koulutetaan mallia...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Ennustetaan testidatasta
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = evaluate(y_test, y_pred, y_prob)

    print("\nTestidatan mittarit:")
    for name, value in metrics.items():
        print(f"{name:<10}: {value:.4f}")
    
    # Tallennetaan artefaktit ja malli
    model_path = os.path.join(MODELS_DIR, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\nMalli tallennettu ({model_path})")

    stats_path = os.path.join(MODELS_DIR, "reference_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(ref_stats, f, indent=4)
    print(f"Tilastot tallennettu ({stats_path})")

    preds_path = os.path.join(MODELS_DIR, "reference_predictions.npy")
    np.save(preds_path, y_pred)
    print(f"Ennusteet tallennettu ({preds_path})")

    labels_path = os.path.join(MODELS_DIR, "validation_labels.npy")
    np.save(labels_path, y_test)
    print(f"Testidata tallennettu ({labels_path})")

    print("\nKoulutusprosessi valmis!")

if __name__ == "__main__":
    train()