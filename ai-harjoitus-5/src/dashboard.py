import matplotlib
matplotlib.use("agg")
import os
import pickle
import json
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from train import DATA_DIR, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET
from monitor import (
    MODEL_PATH, STATS_PATH, REF_PREDS_PATH, VAL_LABELS_PATH, PSI_WARNING_THRESHOLD,
    PSI_ALERT_THRESHOLD, PERFORMANCE_DROP_THRESHOLD, PerformanceMonitor, KSTester, PSICalculator
)

# Väripaletti
BACKGROUND = "#555755"
PANEL = "#474F59"
ACCENT = "#7E9BBD"
ACCENT2 = "#58A36D"
WARNING = "#D48524"
ALERT = "#C91637"
TEXT = "#EDF0F2"
TEXT_MUTED = "#8C8E91"
GRID = "#383D45"

# Sivun asetukset
st.set_page_config(
    page_title="Telco-asiakaspysyvyysmallin monitorointi",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Globaali CSS-tyylitys
st.markdown("""
<style>      
html, body, [class*="css"] {
    background-color: #555755;
    color: #edf0f2;  
    font-family: "sans-serif";  
}
.stTabs [data-baseweb="tab-list"] {
    padding: 6px;
    gap: 4px;
    background: #474f59;
    border-radius: 10px;
}
.stTabs [data-baseweb="tab"] {
    padding: 8px 22px;
    background: transparent;
    color: #8c8e91;
    font-family: "sans-serif";
    font-weight: 700;
    font-size: 14px;
    border-radius: 7px;
    letter-spacing: 0.05em;
    text-transform: uppercase;            
}
.stTabs [aria-selected="true"] {
    background: #7e9bbd;
    color: #edf0f2;
}  
.metric-card {
    margin-bottom: 8px;
    padding: 20px 24px;
    background: #474f59;
    border: 1px solid #383d45;
    border-radius: 12px;
}
.metric-card h3 {
    margin: 0 0 6px 0;
    color: #8c8e91;
    font-family: "monospace";
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
.metric-card .value {
    color: #7e9bbd;
    font-family: "sans-serif";
    font-size: 32px;
    font-weight: 800;
    line-height: 1;
}
.metric-card .sub {
    margin-top: 4px;
    color: #8c8e91;
    font-family: "monospace";
    font-size: 12px;
}
.alert-box {
    margin-bottom: 6px;
    padding: 10px 16px;
    font-family: "monospace";
    font-size: 12px;
    border-radius: 8px;
}
.alert-danger {
    background: #c91637;
    color: #edf0f2;
    border-left: 3px solid #c91637;
}
.alert-warning {
    background: #d48524;
    color: #edf0f2;
    border-left: 3px solid #d48524;
}
.alert-ok {
    background: #58a36d;
    color: #edf0f2;
    border-left: 3px solid #58a36d;
}
.section-header {
    margin: 24px 0 12px 0;
    padding-bottom: 6px;
    color: #7e9bbd;
    font-family: "monospace";
    font-size: 11px;
    border-bottom: 1px solid #383d45;
    text-transform: uppercase;
    letter-spacing: 0.15em;   
}
</style>            
""", unsafe_allow_html=True)

def _apply_plot_styles(fig, ax_list):
    """
    Soveltaa yhtenäiset tyylit Matplotlib-kaavioihin.

    Parametrit:
        fig (matplotlib.figure.Figure): Kaaviokuva.
        ax_list (list[matplotlib.axes.Axes]): Lista kaaviokuvan akseleista.
    """
    fig.patch.set_facecolor(PANEL)
    for ax in ax_list:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT_MUTED, labelsize=8)
        ax.xaxis.label.set_color(TEXT_MUTED)
        ax.yaxis.label.set_color(TEXT_MUTED)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.yaxis.set_tick_params(labelcolor=TEXT_MUTED)
        ax.xaxis.set_tick_params(labelcolor=TEXT_MUTED)
        ax.grid(color=GRID, linestyle="--", linewidth=0.5, alpha=0.6)

def _psi_color(psi_value):
    """
    Palauttaa värin PSI-arvon perusteella.

    Parametrit:
        psi_value (float): PSI-arvo.

    Palauttaa:
        str: Värikoodi.
    """
    if np.isnan(psi_value):
        return TEXT_MUTED
    if psi_value >= PSI_ALERT_THRESHOLD:
        return ALERT
    if psi_value >= PSI_WARNING_THRESHOLD:
        return WARNING
    return ACCENT2

# Välimuisti data ja aftefaktien lataukselle
@st.cache_resource(show_spinner="Ladataan mallia ja tilastoja...")
def load_artifacts():
    """
    Lataa ja palauttaa kaikki seurantaan tarvittavat aftefaktit.

    Palauttaa:
        tuple: (model, reference_stats, reference_preds, validation_labels)
    """
    model, reference_stats, reference_preds, validation_labels = None, None, None, None
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH, "r") as f:
            reference_stats = json.load(f)
    if os.path.exists(REF_PREDS_PATH):
        reference_preds = np.load(REF_PREDS_PATH)
    if os.path.exists(VAL_LABELS_PATH):
        validation_labels = np.load(VAL_LABELS_PATH)
    return model, reference_stats, reference_preds, validation_labels

@st.cache_data(show_spinner="Ladataan dataa...")
def load_data():
    """
    Lataa, esikäsittelee ja jakaa Telco-datan referenssi- ja nykyeriin.

    Palauttaa:
        tuple: (df, X_reference, X_current, y_current)
    """
    df = pd.read_csv(DATA_DIR)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.drop(columns=["customerID"], inplace=True)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    X_reference, X_current = train_test_split(df, test_size=0.2, random_state=99, stratify=df[TARGET])
    y_current = X_current[TARGET].values

    return df, X_reference, X_current, y_current

# Käännöstaulu: avaimet -> näyttötekstit
METRIC_NAMES = {
    "tarkkuus": "Tarkkuus",
    "f1_pisteet": "F1-pisteet",
    "sisäinen_tarkkuus": "Sisäinen tarkkuus",
    "herkkyys": "Herkkyys",
    "roc_auc": "ROC-AUC"
}

def _validate_aftifacts():
    """
    Tarkistaa, että kaikki tarvittavat aftefaktit on ladattu onnistuneesti.

    Palauttaa:
        lis[str]: Lista puuttuvista aftefakteista.
    """
    required = {
        "Malli (model.pkl)": MODEL_PATH,
        "Referenssitilastot (reference_stats.json)": STATS_PATH,
        "Referenssiennusteet (reference_predictions.npy)": REF_PREDS_PATH,
        "Validointilabelit (validation_labels.npy)": VAL_LABELS_PATH,
        "Data (Wa_Fn-UseC_-Telco-Customer-Churn.csv)": DATA_DIR
    }
    missing = [name for name, path in required.items() if not os.path.exists(path)]
    return missing

def render_sidebar(report):
    """
    Renderöi sivupalkin hälytysten yhteenvedolla.

    Parametrit:
        report (dict | None): Monitorointiraportti, joka sisältää PSI-, KS- ja suorituskykymittarit tai None, jos aftefakteja ei ole ladattu.
    """
    with st.sidebar:
        st.markdown(
            "<div style='margin-bottom:4px;color:#8c8e91;font-family:monospace;"
            "font-size:11px;letter-spacing:0.2em;text-transform:uppercase;'>"
            "Telco-asiakaspysyvyysmallin monitorointi</div>"
            "<div sttyle='margin-bottom:20px;color:7e9bbd;font-family:sans-serif;"
            "font-size:23px;font-weight:800px;'>Monitoroi</div>",
            unsafe_allow_html=True
        )

        if report is None:
            st.info("Suoraita train.py ensin aftefaktien luomiseksi")
            return
        
        alerts = report.get("alerts", [])
        n_alerts = sum(1 for a in alerts if a["severity"] == "hälytys")
        n_warnings = sum(1 for a in alerts if a["severity"] == "varoitus")

        st.markdown("<div class='section-header'>Hälytysyhteenveto</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        col1.metric("Hälytykset", n_alerts)
        col2.metric("Varoitukset", n_warnings)

        if alerts:
            st.markdown("<div class='section-header'>Aktiiviset hälytykset</div>", unsafe_allow_html=True)
            for alert in alerts:
                css = "alert-danger" if alert["severity"] == "hälytys" else "alert-warning"
                text = "[HÄLYTYS]" if alert["severity"] == "hälytys" else "[VAROITUS]"
                st.markdown(
                    f"<div class='alert-box {css}'>{text} {alert['message']}</div>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                "<div class='alert-box alert-ok'>Ei aktiivisia hälytyksiä</div>",
                unsafe_allow_html=True
            )

def render_overview(df, X_reference, X_current, y_current, reference_preds, validation_labels):
    """
    Renderöi yleiskatsausvälilehden sisällön.

    Parametrit:
        df (pd.DataFrame): Koko esikäsitelty DataFrame.
        X_reference (pd.DataFrame): Referenssierä.
        X_current (pd.DataFrame): Nykyinen seurantaerä.
        y_current (np.ndarray): Todelliset luokat nykyisessä erässä.
        reference_preds (np.ndarray): Referenssierän ennusteet.
        validation_labels (np.ndarray): Validointilabelit.
    """
    st.markdown("<div class='section-header'>Aineiston yleiskatsaus</div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    total_customers = len(df)
    churn_rate = df[TARGET].mean() * 100
    refernce_size = len(X_reference)
    current_size = len(X_current)

    for col, label, value, sub in [
        (col1, "Asiakkaita yhteensä", f"{total_customers:,}", "koko aineisto"),
        (col2, "Pysyvyysprosentti", f"{churn_rate:.1f}%", "kaikista asiakkaista"),
        (col3, "Refernssierä", f"{refernce_size:,}", "80% aineistosta"),
        (col4, "Nykyinen seurantaerä", f"{current_size:,}", "20% aineistosta")
    ]:
        col.markdown(
            f"<div class='metric-card'><h3>{label}</h3>"
            f"<div class='value'>{value}</div>"
            f"<div class='sub'>{sub}</div></div>",
            unsafe_allow_html=True
        )
    
    # Referenssin suorituskykymittarit
    if reference_preds is not None and validation_labels is not None:
        st.markdown("<div class='section-header'>Referenssin suorituskykymittarit (validointidata)</div>", unsafe_allow_html=True)
        reference_metrics = {METRIC_NAMES[key]: value
                             for key, value in PerformanceMonitor().compute_metrics(validation_labels, reference_preds).items()}
        metric_cols = st.columns(len(reference_metrics))
        for (name, value), metric_col in zip(reference_metrics.items(), metric_cols):
            metric_col.markdown(
                f"<div class='metric-card'><h3>{name}</h3>"
                f"<div class='value' style='font-size:24px;'>{value:.4f}</div></div>",
                unsafe_allow_html=True
            )

    st.markdown("<div class='section-header'>Datajakumat: referenssi vs. nykyinen erä</div>", unsafe_allow_html=True)

    # Histogrammit numeerisille ominaisuuksille
    fig, axes = plt.subplots(1, len(NUMERIC_FEATURES), figsize=(14, 3))
    for ax, feature in zip(axes, NUMERIC_FEATURES):
        reference_values = X_reference[feature].dropna()
        current_values = X_current[feature].dropna()
        ax.hist(reference_values, bins=25, alpha=0.6, color=ACCENT, label="Referenssi", density=True)
        ax.hist(current_values, bins=25, alpha=0.6, color=ACCENT2, label="Nykinen", density=True)
        ax.set_title(feature, fontsize=9, color=TEXT, fontweight="bold")
        ax.set_xlabel("Arvo", fontsize=7)
        ax.set_ylabel("Tiheys", fontsize=7)
    axes[0].legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT_MUTED)
    _apply_plot_styles(fig, axes)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Pylväsdiagrammit kategoriselle esimerkkiominaisuudelle "Contract"
    st.markdown("<div class='section-header'>Kategorinen jakauma: Sopimus-tyyppi</div>", unsafe_allow_html=True)

    fig2, ax2 = plt.subplots(figsize=(8, 3))
    categories = sorted(df["Contract"].dropna().unique())
    reference_counts = X_reference["Contract"].value_counts(normalize=True).reindex(categories, fill_value=0)
    current_counts = X_current["Contract"].value_counts(normalize=True).reindex(categories, fill_value=0)
    x = np.arange(len(categories))
    w = 0.35
    ax2.bar(x - w/2, reference_counts.values, width=w, color=ACCENT, alpha=0.85, label="Referenssi")
    ax2.bar(x + w/2, current_counts.values, width=w, color=ACCENT2, alpha=0.85, label="Nykyinen")
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=8)
    ax2.set_ylabel("Prosenttiosuus", fontsize=8)
    ax2.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT_MUTED)
    _apply_plot_styles(fig2, [ax2])
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

def render_drift(psi_results, ks_results, reference_stats, X_current):
    """
    Renderöi ajautumisanalyysivälilehden PSI- ja KS-tulokset.

    Parametrit:
        psi_results (dict): PSI-arvot
        ks_results (dict): KS-tulokset
        reference_stats (dict): Referenssitilastot
        X_current (pd.DataFrame): Nykyinen seurantaerä.
    """
    st.markdown("<div class='section-header'>PSI (Population Stability Index)</div>", unsafe_allow_html=True)

    # PSI-pylväsdiagrammi ominaisuuksittain
    sorted_psi = sorted(psi_results.items(), key=lambda x: (np.nan if np.isnan(x[1]) else x[1]), reverse=True)
    features = [key for key, _ in sorted_psi]
    values = [value if not np.isnan(value) else 0 for _, value in sorted_psi]
    colors = [_psi_color(value) for _, value in sorted_psi]

    fig, ax = plt.subplots(figsize=(14, 4))
    bars = ax.barh(features, values, color=colors, edgecolor="none", height=0.6)
    ax.axvline(PSI_WARNING_THRESHOLD, color=WARNING, linestyle="--", linewidth=1, label=f"Varoitusraja ({PSI_WARNING_THRESHOLD})")
    ax.axvline(PSI_ALERT_THRESHOLD, color=ALERT, linestyle="--", linewidth=1, label=f"Hälytysraja ({PSI_ALERT_THRESHOLD})")
    ax.set_xlabel("PSI-arvo", fontsize=8)
    ax.set_title("PSI-arvot ominaisuuksittain", fontsize=10, fontweight="bold", color=TEXT)
    ax.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT_MUTED)
    ax.invert_yaxis()
    # Lisätään arvot palkkien päälle
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{value:.4f}", va="center", ha="left", fontsize=7, color=TEXT_MUTED, 
                fontfamily="monospace")
    _apply_plot_styles(fig, [ax])
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Legendaselitys
    lc1, lc2, lc3 = st.columns(3)
    lc1.markdown(f"<div class='alert-box alert-ok'>[OK] PSI < {PSI_WARNING_THRESHOLD} - Ei muutosta</div>", unsafe_allow_html=True)
    lc2.markdown(f"<div class='alert-box alert-warning'>[VAROITUS] {PSI_WARNING_THRESHOLD} <= PSI < {PSI_ALERT_THRESHOLD} - Lievä muutos</div>", unsafe_allow_html=True)
    lc3.markdown(f"<div class='alert-box alert-danger'>[HÄLYTYS] PSI >= {PSI_ALERT_THRESHOLD} - Merkittävä muutos</div>", unsafe_allow_html=True)

    # KS-testi
    st.markdown("<div class='section-header'>Kolmogorov-Smirnow-testi (numeeriset ominaisuudet)</div>", unsafe_allow_html=True)

    ks_cols = st.columns(len(ks_results))
    for col, feature in zip(ks_cols, NUMERIC_FEATURES):
        result = ks_results.get(feature, {})
        if not result:
            col.markdown(
                f"<div class='metric-card'><h3>{feature}</h3><div class='sub'>Ei tuloksia</div></div>",
                unsafe_allow_html=True
            )
            continue
        drift = result.get("drift_detected", False)
        color = ALERT if drift else ACCENT2
        text = "HÄLYTYS" if drift else "OK"
        col.markdown(
            "<div class='metric-card'>"
            f"<h3>{feature}</h3>"
            f"<div class='value' style='color:{color};font-size:20px'>{text} {'Ajautuminen' if drift else 'Vakaa'}</div>"
            f"<div class='sub'>KS-testi={result['ks_statistic']:.4f} | p-arvo={result['p_value']:.4f}</div>"
            "</div>",
            unsafe_allow_html=True
        )

    # CDF-vertailukaaviot numeerisille ominaisuuksille
    st.markdown("<div class='section-header'>Empiiriset kertymäfunktiot (ECDF)</div>", unsafe_allow_html=True)

    fig2, axes = plt.subplots(1, len(NUMERIC_FEATURES), figsize=(14, 3))
    for ax, feature in zip(axes, NUMERIC_FEATURES):
        if feature not in reference_stats:
            continue
        reference_samples = KSTester(reference_stats).reconstruct_samples_from_histogram(reference_stats[feature]["histogram"])
        current_samples = X_current[feature].dropna().values
        # ECDF-referenssi
        reference_sorted = np.sort(reference_samples)
        reference_ecdf = np.arange(1, len(reference_sorted) + 1) / len(reference_sorted)
        # ECDF nykyinen
        current_sorted = np.sort(current_samples)
        current_ecdf = np.arange(1, len(current_sorted) + 1) / len(current_sorted)
        ax.plot(reference_sorted, reference_ecdf, color=ACCENT, linewidth=1.5, label="Referenssi")
        ax.plot(current_sorted, current_ecdf, color=ALERT, linewidth=1.5, label="Nykyinen")
        ax.set_title(feature, fontsize=9, color=TEXT, fontweight="bold")
        ax.set_xlabel("Arvo", fontsize=7)
        ax.set_ylabel("ECDF", fontsize=7)
    axes[0].legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT_MUTED)
    _apply_plot_styles(fig2, axes)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # PSI-taulukko
    st.markdown("<div class='section-header'>PSI-taulukko</div>", unsafe_allow_html=True)
    psi_df = pd.DataFrame([
        {
            "Ominaisuus": feature,
            "PSI-arvo": f"{value:.4f}" if not np.isnan(value) else "N/A",
            "Tila": ("hälytys" if value >= PSI_ALERT_THRESHOLD else
                     "varoitus" if value >= PSI_WARNING_THRESHOLD else
                     "ok") if not np.isnan(value) else "N/A"
        }
        for feature, value in sorted(psi_results.items(), key=lambda x: x[0])
    ])
    st.dataframe(psi_df, width="stretch", hide_index=True)

def render_performance(model, X_current, y_current, reference_preds, validation_labels):
    """
    Renderöi suorituskykyanalyysinvälidelehden mittarit ja vertailut.

    Parametrit:
        model (sklearn.pipeline.Pipeline | None): Koulutettu ja ladattu malli.
        X_current (pd.DataFrame): Nykyinen seurantaerä.
        y_current (np.ndarray): Todelliset luokat nykyisessä erässä.
        reference_preds (np.ndarray): Referenssierän ennusteet.
        validation_labels (np.ndarray): Validointilabelit. 
    """
    if model is None:
        st.warning("Mallia ei löydetty, suorita train.py ensin")
        return
    
    # Ennustetaan nykyisestä erästä
    X_features = X_current.drop(columns=[TARGET])
    y_prob = model.predict_proba(X_features)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    _perf_monitor = PerformanceMonitor()
    current_metrics = {METRIC_NAMES[key]: value
                       for key, value in _perf_monitor.compute_metrics(y_current, y_pred, y_prob).items()}
    reference_metrics = {METRIC_NAMES[key]: value
                         for key, value in _perf_monitor.compute_metrics(validation_labels, reference_preds).items()} if (
                             reference_preds is not None and validation_labels is not None
                         ) else {}
    
    st.markdown("<div class='section-header'>Suorituskykymittarit (nykyinen erä vs. refernssierä)</div>", unsafe_allow_html=True)

    metric_cols = st.columns(len(current_metrics))
    for (name, current_value), metric_col in zip(current_metrics.items(), metric_cols):
        reference_value = reference_metrics.get(name, np.nan)
        if not np.isnan(reference_value):
            delta = current_value - reference_value
            alert = delta < -PERFORMANCE_DROP_THRESHOLD
            delta_color = ALERT if alert else (ACCENT2 if delta >= 0 else WARNING)
            delta_str = f"{delta:+.4f}"
        else:
            delta_color = TEXT_MUTED
            delta_str = "—"

        reference_value_str = f"{reference_value:.4f}" if not np.isnan(reference_value) else "N/A"
        metric_col.markdown(
            "<div class='metric-card'>"
            f"<h3>{name}</h3>"
            f"<div clasS='value' style='font-size:22px;'>{current_value:.4f}</div>"
            f"<div class='sub'>Referenssi: {reference_value_str} "
            f"| <span style='color:{delta_color};'>{delta_str}</span></div>"
            "</div>",
            unsafe_allow_html=True
        )

    # Suorituskykyvertailukaaviot
    st.markdown("<div class='section-header'>Mittariverailu</div>", unsafe_allow_html=True)

    shared_metrics = [key for key in current_metrics if key in reference_metrics]
    if shared_metrics:
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(shared_metrics))
        w = 0.35
        current_values = [current_metrics[metric] for metric in shared_metrics]
        reference_values = [reference_metrics[metric] for metric in shared_metrics]
        ax.bar(x - w/2, reference_values, width=w, color=ACCENT, alpha=0.85, label="Referenssi")
        ax.bar(x + w/2, current_values, width=w, color=ACCENT2, alpha=0.85, label="Nykyinen")
        ax.set_xticks(x)
        ax.set_xticklabels(shared_metrics, fontsize=9, rotation=15, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Arvo", fontsize=8)
        ax.legend(fontsize=9, facecolor=PANEL, labelcolor=TEXT_MUTED)
        ax.set_title("Suorituskykymittarit: referenssi vs. nykyinen", fontsize=10, color=TEXT)
        _apply_plot_styles(fig, [ax])
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    # Ennusteen todennäköisyysjakauma
    st.markdown("<div class='section-header'>Ennusteen todennäköisyysjakauma (nykyinen erä)</div>", unsafe_allow_html=True)

    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.hist(y_prob[y_current == 0], bins=40, alpha=0.6, color=ACCENT2, label="Ei pysynyt asiakkaana (0)", density=True)
    ax.hist(y_prob[y_current == 1], bins=40, alpha=0.6, color=ALERT, label="Pysyi asiakkaana (1)", density=True)
    ax.axvline(0.5, color=TEXT_MUTED, linestyle="--", linewidth=1, label="Kynnysarvo 0.5")
    ax2.set_xlabel("Ennusteen todennäköisyys pysyvyydelle", fontsize=8)
    ax2.set_ylabel("Tiheys", fontsize=8)
    ax2.set_title("Ennusteen todennäköisyysjakauma nykyisessä erässä", fontsize=10, color=TEXT)
    ax2.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT_MUTED)
    _apply_plot_styles(fig2, [ax2])
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # Sekaannusmatriisi
    st.markdown("<div class='section-header'>Sekaannusmatriisi (nykyinen erä)</div>", unsafe_allow_html=True)

    cm = confusion_matrix(y_current, y_pred)  
    fig3, ax3 = plt.subplots(figsize=(4, 3))
    im = ax3.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color=TEXT, fontsize=14, fontweight="bold")
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(["Ei pysynyt (0)", "Pysyi (1)"], fontsize=9, color=TEXT_MUTED)
    ax3.set_yticklabels(["Ei pysynyt (0)", "Pysyi (1)"], fontsize=9, color=TEXT_MUTED)
    ax3.set_xlabel("Ennustettu luokka", fontsize=8)
    ax3.set_ylabel("Todellinen luokka", fontsize=8)
    ax3.set_title("Sekaannusmatriisi nykyisessä erässä", fontsize=9, color=TEXT)
    fig3.patch.set_facecolor(PANEL)
    ax3.set_facecolor(PANEL)
    for spine in ax3.spines.values():
        spine.set_edgecolor(GRID)
    plt.colorbar(im, ax=ax3)
    fig3.tight_layout()
    col_cm, _ = st.columns([1, 2])
    col_cm.pyplot(fig3)
    plt.close(fig3)

def main():
    """
    Streamlit-sovelluksen pääfunktio.
    """
    # Validointitarkistus
    missing = _validate_aftifacts()
    if missing:
        st.error(
            "**Puuttuvat artefaktit**, suorita `python run_all.py ` ensin:\n"
            + "\n".join(f"- {m}" for m in missing)
        )
        st.stop()

    # Ladataan data ja aftefaktit
    model, reference_stats, reference_preds, validation_labels = load_artifacts()
    df, X_reference, X_current, y_current = load_data()

    # Lasketaan PSI ja KS-tulokset
    X_current_feature = X_current.drop(columns=[TARGET])
    _psi_calculator = PSICalculator(reference_stats)
    _ks_tester = KSTester(reference_stats)
    psi_results = _psi_calculator.calculate_all_psi(X_current_feature, NUMERIC_FEATURES, CATEGORICAL_FEATURES) if reference_stats else {}
    ks_results = _ks_tester.perform_all_ks_test(X_current_feature, NUMERIC_FEATURES) if reference_stats else {}

    # Kotaann raportti sivupalkkiin
    alerts = []
    for feature, value in psi_results.items():
        if np.isnan(value):
            continue
        if value >= PSI_ALERT_THRESHOLD:
            alerts.append({"severity": "hälytys", "message": f"PSI-hälytys: {feature} (PSI={value:.4f})"})
        elif value >= PSI_WARNING_THRESHOLD:
            alerts.append({"severity": "varoitus", "message": f"PSI-varoitus: {feature} (PSI={value:.4f})"})

    for feature, result in ks_results.items():
        if result.get("drift_detected"):
            alerts.append({"severity": "hälytys", "message": f"KS-hälytys: {feature} (p-arvo={result['p_value']:.4f})"})
    
    report = {"alerts": alerts}

    render_sidebar(report)

    # Välilehdet
    tab1, tab2, tab3 = st.tabs(["Yleiskatsaus", "Ajautumisanalyysi", "Suorituskyky"])

    with tab1:
        render_overview(df, X_reference, X_current, y_current, reference_preds, validation_labels)

    with tab2:
        if reference_stats:
            render_drift(psi_results, ks_results, reference_stats, X_current_feature)
        else:
            st.info("Referenssitilastoja ei löydy, suorita train.py ensin")

    with tab3:
        render_performance(model, X_current, y_current, reference_preds, validation_labels)

if __name__ == "__main__":
    main()