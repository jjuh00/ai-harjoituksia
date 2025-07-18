"""
Pääohjelma opiskelijoiden suoritusten ennustamiseen ja vinouman analysointiin.

Ohjelma käyttää predictor.py-moduulia, joka sisältää datan esikäsittelyn, mallien koulutuksen ja arvioinnin,
opiskelijoiden suoritusten ennustamiseen sekä vinouman analysointiin Fairlearn-kirjaston avulla.
"""

import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    MetricFrame
)
from sklearn.metrics import accuracy_score, precision_score, recall_score
from predictor import StudentPerformancePredictor

warnings.filterwarnings("ignore")

plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.titlesize"] = 16

def load_original_data():
    """
    Lataa alkuperäisen datan visualisoimista varten.
    
    Palauttaa:
        pd.DataFrame: Alkuperäinen data
    """
    try:
        data = pd.read_csv("StudentsPerformance.csv")
        # Luodaan pass-sarake (sama logiikka kuin predictor.py:ssä)
        avg_score = (data["math score"] + data["reading score"] + data["writing score"]) / 3
        data["pass"] = (avg_score >= 50).astype(int)
        return data
    except FileNotFoundError:
        print("Tiedostoa ei löytynyt. Varmista, että se on oikeassa kansiossa (ai-harjoitus-3)")
        return None
    except Exception as e:
        print(f"Virhe datan lataamisessa: {e}")
        return None
    
def visualize_original_data(data):
    """
    Visualisoi alkuperäisen datan jakaumat.

    Parametrit:
        data (pd.DataFrame): Alkuperäinen data, joka sisältää opiskelijoiden suoritukset
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Alkuperäisen datan jakaumat", fontsize=16, fontweight="bold")

    # Sukupuolijakauma
    gender_counts = data["gender"].value_counts()
    axes[0, 0].pie(gender_counts.values, labels=gender_counts.index, autopct="%1.1f%%",
                   colors=["deeppink", "mediumslateblue"])
    axes[0, 0].set_title("Sukupuolijakauma")

    # Etnisen taustan jakauma
    race_counts = data["race/ethnicity"].value_counts()
    axes[0, 1].bar(range(len(race_counts)), race_counts.values, color="seagreen")
    axes[0, 1].set_title("Etnisen taustan jakauma")
    axes[0, 1].set_xlabel("Etninen tausta")
    axes[0, 1].set_ylabel("Lukumäärä")
    axes[0, 1].set_xticks(range(len(race_counts)))
    axes[0, 1].set_xticklabels(race_counts.index, rotation=45, ha="right")

    # Läpi päässeiden jakauma
    pass_counts = data["pass"].value_counts()
    axes[0, 2].pie(pass_counts.values, labels=["Ei läpäissyt", "Läpäisi"], autopct="%1.1f%%",
                   colors=["orangered", "limegreen"])
    axes[0, 2].set_title("Läpi päässeiden jakauma")

    # Pisteiden jakauma
    scores = ["math score", "reading score", "writing score"]
    score_titles = ["Matikan pisteet", "Lukutaidon pisteet", "Kirjoitustaidon pisteet"]
    colors = ["cornflowerblue", "gold", "crimson"]

    for i, (score, title, color) in enumerate(zip(scores, score_titles, colors)):
        axes[1, i].hist(data[score], bins=20, alpha=0.7, color=color, edgecolor="black")
        axes[1, i].set_title(title)
        axes[1, i].set_xlabel("Pisteet")
        axes[1, i].set_ylabel("Lukumäärä")
        axes[1, i].axvline(x=50, color="dimgray", linestyle="--", label="Läpäisyraja (50)")
        axes[1, i].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.4)
    plt.show()

def visualize_model_performances(results_df):
    """
    Visualisoi mallien suorituskyvyn vertailun.

    Parametrit:
        results_df (pd.DataFrame): Mallien suorituskyvyn tulokset
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Mallien suorituskyvyn vertailu", fontsize=16, fontweight="bold")

    metrics = ["Tarkkuus", "Tarkkuus (sisäinen)", "Herkkyys", "F1-arvo"]
    colors = ["royalblue", "springgreen", "salmon", "yellow"]
    x_pos = np.arange(len(results_df))
    width = 0.2
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        axes[0].bar(x_pos + i * width, results_df[metric], width, label=metric, color=color, alpha=0.8)

    # Pylväsdiagrammi mallien suorituskyvystä
    axes[0].set_xlabel("Mallit")
    axes[0].set_ylabel("Arvo")
    axes[0].set_title("Mallien suorituskyky")
    axes[0].set_xticks(x_pos + width * 1.5)
    axes[0].set_xticklabels(results_df["Malli"], rotation=45, ha="right")
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)

    # Lämpökartta suorituskyvyn vertailusta
    performance_matrix = results_df.set_index("Malli")[metrics].T
    sns.heatmap(performance_matrix, annot=True, fmt=".3f", cmap="RdYlGn",
                cbar_kws={"label": "Suorituskyky"}, ax=axes[1])
    axes[1].set_title("Mallien suorituskyvyn vertailun lämpökartta")
    axes[1].set_xlabel("Mallit")
    axes[1].set_ylabel("Mittarit")

    plt.tight_layout()
    plt.show()

def analyze_bias(predictor, model_name="Satunnaismetsä"):
    """
    Analysoi vinoumaa mallin ennusteissa Fairlearn-kirjaston avulla.
    
    Parametrit:
        predictor (StudentPerformancePredictor): Ennustaja, joka sisältää mallin tulokset
        model_name (str): Mallin nimi, jota käytetään vinouman analysointiin

    Palauttaa:
        dict: Vinouman analyysin tulokset
    """
    # Haetaan ennustukset ja herkät ominaisuudet
    predictions, sensitive_features_test, y_test = predictor.get_model_predictions_for_bias_analysis(model_name)

    # Luodaan vinouma-analyysi sukupuolen perusteella
    gender_bias = {}

    # Väestöllinen tasa-arvo
    gender_dp = demographic_parity_difference(y_test, predictions, sensitive_features=sensitive_features_test["gender"])

    # Tasavertaiset kertoimet
    gender_eo = equalized_odds_difference(y_test, predictions, sensitive_features=sensitive_features_test["gender"])

    gender_bias["demographic_parity"] = gender_dp
    gender_bias["equalized_odds"] = gender_eo

    # Luodaan vinouma-analyysi etnisen taustan perusteella
    race_bias = {}

    # Väestöllinen tasa-arvo
    race_dp = demographic_parity_difference(y_test, predictions, sensitive_features=sensitive_features_test["race/ethnicity"])

    # Tasavertaiset kertoimet
    race_eo = equalized_odds_difference(y_test, predictions, sensitive_features=sensitive_features_test["race/ethnicity"])

    race_bias["demographic_parity"] = race_dp
    race_bias["equalized_odds"] = race_eo

    # Luodaan MetricFrame (sukupuolen) vinouman analysointiin
    metric_frame_gender = MetricFrame(
        metrics={"Tarkkuus": accuracy_score, "Tarkkuus (sisäinen)": precision_score, "Herkkyys": recall_score},
        y_true=y_test,
        y_pred=predictions,
        sensitive_features=sensitive_features_test["gender"]
    )

    # Luodaan MetricFrame (etnisen taustan) vinouman analysointiin
    metric_frame_race = MetricFrame(
        metrics={"Tarkkuus": accuracy_score, "Tarkkuus (sisäinen)": precision_score, "Herkkyys": recall_score},
        y_true=y_test,
        y_pred=predictions,
        sensitive_features=sensitive_features_test["race/ethnicity"]
    )

    return {
        "gender_bias": gender_bias,
        "race_bias": race_bias,
        "gender_bias_metric_frame": metric_frame_gender,
        "race_bias_metric_frame": metric_frame_race,
        "model_name": model_name
    }
    
def visualize_bias_analysis(bias_results):
    """
    Visualisoi vinouman analyysin tulokset.
    
    Parametrit:
        bias_results (dict): Vinouman analyysin tulokset
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f"Vinouman analyysi: {bias_results['model_name']}", fontsize=16, fontweight="bold")

    # Sukupuolen vinouman visualisointi
    bias_metrics = ["demographic_parity", "equalized_odds"]
    gender_values = [bias_results["gender_bias"][metric] for metric in bias_metrics]

    bars1 = axes[0, 0].bar(bias_metrics, gender_values, color=["seagreen", "mediumaquamarine"], alpha=0.8)
    axes[0, 0].set_title("Sukupuolivinouma")
    axes[0, 0].set_ylabel("Vinouman arvo")
    axes[0, 0].set_xticklabels(["Väestöllinen\ntasa-arvo", "Tasavertaiset\nkertoimet"])
    axes[0, 0].axhline(y=0, color="black", linestyle="--", alpha=0.3)
    axes[0, 0].grid(True, alpha=0.3)

    # Lisätään arvot pylväiden päälle
    for bar, value in zip(bars1, gender_values):
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{value:.3f}", ha="center", va="bottom")
        
    # Etnisen taustan vinouma
    race_values = [bias_results["race_bias"][metric] for metric in bias_metrics]

    bars2 = axes[0, 1].bar(bias_metrics, race_values, color=["seagreen", "mediumaquamarine"], alpha=0.8)
    axes[0, 1].set_title("Etnisyysvinouma")
    axes[0, 1].set_ylabel("Vinouman arvo")
    axes[0, 1].set_xticklabels(["Väestöllinen\ntasa-arvo", "Tasavertaiset\nkertoimet"])
    axes[0, 1].axhline(y=0, color="black", linestyle="--", alpha=0.3)
    axes[0, 1].grid(True, alpha=0.3)

    # Lisätään arvot pylväiden päälle
    for bar, value in zip(bars2, race_values):
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{value:.3f}", ha="center", va="bottom")
        
    # Sukupuolen suorituskykyvertailu
    gender_metrics_df = bias_results["gender_bias_metric_frame"].by_group
    gender_metrics_df.plot(kind="bar", ax=axes[1, 0], alpha=0.8)
    axes[1, 0].set_title("Suorituskyky sukupuolen mukaan")
    axes[1, 0].set_ylabel("Arvo")
    axes[1, 0].set_xlabel("Sukupuoli")
    axes[1, 0].set_xticklabels(["female", "male"]) # Enkoodattu tässä muodossa/järjestyksessä
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=0)

    # Etnisen taustan suorituskykyvertailu
    race_metrics_df = bias_results["race_bias_metric_frame"].by_group
    race_metrics_df.plot(kind="bar", ax=axes[1, 1], alpha=0.8)
    axes[1, 1].set_title("Suorituskyky etnisyyden mukaan")
    axes[1, 1].set_ylabel("Arvo")
    axes[1, 1].set_xlabel("Etninen ryhmä")
    axes[1, 1].set_xticklabels(["group A", "group B", "group C", "group D", "group E"]) # Enkoodattu tässä muodossa/järjestyksessä
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.subplots_adjust(hspace=0.4)
    plt.show()

def _interpret_bias_value(value, bias_type):
    """
    Tulkitsee vinouman arvon.
    
    Parametrit:
        value (float): Vinouman arvo
        bias_type (str): Vinouman tyyppi
    """
    abs_value = abs(value)
    if abs_value < 0.05:
        return "Vähäinen vinouma"
    elif abs_value < 0.1:
        return "Kohtalainen vinouma"
    else:
        return "Merkittävä vinouma"
    
def print_summary(results_df, bias_results):
    """
    Tulostaa yhteenvedon mallin suorituskyvystä ja vinouma-analyysistä.

    Parametrit:
        results_df (pd.DataFrame): Mallien arvioinnit
        bias_results (dict): Vinouma-analyysin tulokset
    """
    print("Yhteenveto mallien suorituskyvystä ja vioumasta:")

    print("\nMallien suorituskyky (suurempi luku parempi):")
    best_model = results_df.loc[results_df["Tarkkuus"].idxmax()]
    print(f"Paras malli: {best_model['Malli']}")
    print(f"Tarkkuus: {best_model['Tarkkuus']:.3f}")
    print(f"Sisäinen tarkkuus: {best_model['Tarkkuus (sisäinen)']:.3f}")
    print(f"Herkkyys: {best_model['Herkkyys']:.3f}")
    print(f"F1-arvo: {best_model['F1-arvo']:.3f}")

    print("\nKaikkien mallien tarkkuudet (suurempi luku parempi):")
    for _, row in results_df.iterrows():
        print(f"{row['Malli']}: {row['Tarkkuus']:.3f}")

    print("\nVinouman analyysi:")
    print(f"Sukupuolen vinouma (pienempi luku parempi):")
    dp_gender = bias_results["gender_bias"]["demographic_parity"]
    eo_gender = bias_results["gender_bias"]["equalized_odds"]
    print(f"Väestöllinen tasa-arvo: {dp_gender:.3f}")
    print(f"Tasavertaiset kertoimet: {eo_gender:.3f}")

    print("\nEtnisen taustan vinouma (pienempi luku parempi):")
    dp_race = bias_results["race_bias"]["demographic_parity"]
    eo_race = bias_results["race_bias"]["equalized_odds"]
    print(f"Väestöllinen tasa-arvo: {dp_race:.3f}")
    print(f"Tasavertaiset kertoimet: {eo_race:.3f}")

    print("\nVinouman tulkinta:")
    print(f"Sukupuolen vinouma: {_interpret_bias_value(dp_gender, 'demographic_parity')}")
    print(f"Etnisen taustan vinouma: {_interpret_bias_value(dp_race, 'demographic_parity')}")

def main():
    """
    Pääohjelma, joka suorittaa opiskelijoiden suoritusten ennustamisen ja vinouman analyysin.
    """
    predictor = StudentPerformancePredictor()

    # Vaihe 1: Ladataan alkuperäinen data ja visualisoidaan jakaumat
    print("Ladataan ja visualisoidaan alkuperäistä dataa")
    original_data = load_original_data()
    if original_data is None:
        return
    
    visualize_original_data(original_data)

    # Vaihe 2: Ladataan data malleja varten
    print("\nEsikäsitellään dataa")
    
    if not predictor.load_and_prepare_data():
        return
    
    # Vaihe 3. Opetetaan ja arvioidaan mallit
    print("\nOpetetaan ja arvioidaan koneoppimismalleja")
    predictor.train_and_evaluate_models()
    results_df = predictor.get_model_performances()
    visualize_model_performances(results_df)

    # Vaihe 4: Analysoidaan vinoumaa
    print("\nAnalysoidaan vinoumaa malleissa")
    bias_results = analyze_bias(predictor, "Satunnaismetsä")

    visualize_bias_analysis(bias_results)

    # Vaihe 5: Tulostetaan yhteenveto
    print_summary(results_df, bias_results)

if __name__ == "__main__":
    main()