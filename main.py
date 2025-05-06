"""
Ohjelma, joka kertoo/ennustaa Suomen tai 10 suurimman kaupungin väkiluvun
vuosina 2010-2024 hyödyntämällä väestödataa ja koneoppimista.

Ohjelma esikättelee datan, rakentaa neuroverkkomallin ja luo ennusteet sekä visualisoinnit.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from tf_keras.models import Sequential
from tf_keras.layers import Dense, LSTM, Dropout
from tf_keras.optimizers import Adam
from tf_keras.callbacks import EarlyStopping

DATASETS_FOLDER = "aineisto"
CITY_FILE = "vuosikirja_2024_vaesto_04.xlsx"
BIRTH_DEATH_FILE = "vuosikirja_2024_vaesto_32.xlsx"
POPULATION_AGE_FILE = "vuosikirja_2024_vaesto_08.xlsx"
MIGRATION_FILE = "vuosikirja_2024_vaesto_45.xlsx"

def load_city_data():
    """
    Lataa ja käsittelee kaupunkien datan

    Palauttaa:
        DataFrame: Käsitelty kaupunkien väestödata (vuodet sarakkeina ja kaupungit riveinä)
    """
    city_path = os.path.join(DATASETS_FOLDER, CITY_FILE)

    # Luetaan Excel-tiedosto, ohitetaan ensimmäiset 10 riviä
    city_df = pd.read_excel(city_path, skiprows=10)

    # Säilytetään vaan olennaiset rivit (top 50 kaupungit) ja tiputetaan viimeiset 3 riviä
    city_df = city_df.iloc[:50]
    city_df = city_df.dropna()

    # Erotellaan kaupnkien nimet
    cities = city_df.iloc[:, 0].tolist()

    # Haetaan vuodet
    years_df = pd.read_excel(city_path, nrows=1, skiprows=9)
    years = years_df.iloc[0, 1:8].tolist()

    # Erotellaan väestödata
    population_data = city_df.iloc[:, 1:8].values

    # Luodaan siistitty DataFrame (vuodet sarakkeina)
    city_population_df = pd.DataFrame(population_data, index=cities, columns=years)

    # Siistitään dataa (muunnetaan desimaaliluvut kokonaisluvuiksi)
    for col in city_population_df.columns:
        city_population_df[col] = city_population_df[col].astype(float).astype(int)

    return city_population_df

def load_finland_population_data():
    """
    Lataa ja käsittelee Suomen kokonaisväkiluvun

    Palauttaa:
        DataFrame: Suomen väkiluku vuosittain
    """
    # Ladataan väestö ikäryhmittäin (Excelistä)
    population_age_path = os.path.join(DATASETS_FOLDER, POPULATION_AGE_FILE)
    population_df = pd.read_excel(population_age_path, skiprows=9)
    population_df = population_df.dropna(subset=[population_df.columns[0], population_df.columns[1]]) # Suodatetaan pois epäolennaiset sarakkeet
    population_df = population_df.iloc[:27] # Otetaan huomioon vain vuodet (vuoteen) 2023 saakka

    # Erotellaan vuodet ja yhteisväkiluku
    years = population_df.iloc[:, 0].tolist()
    population = population_df.iloc[:, 1].tolist()

    # Luodaan siistitty DataFrame
    finland_population_df = pd.DataFrame({"Vuosi": years, "Väkiluku": population})

    # Muutetaan DataFrameen mahdolliset desimaaliluvut kokonaisluvuiksi
    finland_population_df["Vuosi"] = finland_population_df["Vuosi"].astype(int)
    finland_population_df["Väkiluku"] = finland_population_df["Väkiluku"].astype(int)

    return finland_population_df

def load_birth_death_data():
    """
    Lataa elävänä syntyneiden ja kuolleiden datan täydentävää analysointia varten

    Palauttaa:
        DataFrame: Syntyneet ja kuolleet vuosittain
    """
    bd_path = os.path.join(DATASETS_FOLDER, BIRTH_DEATH_FILE)

    # Luetaan Excelit
    bd_df = pd.read_excel(bd_path, skiprows=238) # Otetaan huomioon vain vuodet 1980 jälkeen
    bd_df = bd_df.dropna() 
    bd_df = bd_df.iloc[:44]

    # Erotellaan vuodet, syntyneet ja kuolleet
    years = bd_df.iloc[:, 0].tolist()
    births = bd_df.iloc[:, 1].tolist()
    deaths = bd_df.iloc[:, 2].tolist()

    # Luodaan siistitty DataFrame
    birth_death_df = pd.DataFrame({"Vuosi": years, "Syntyneet": births, "Kuolleet": deaths})
    birth_death_df = birth_death_df.dropna()

    # Muutetaan DataFrameen mahdolliset desimaaliluvut kokonaisluvuiksi
    birth_death_df["Vuosi"] = birth_death_df["Vuosi"].astype(int)
    birth_death_df["Syntyneet"] = birth_death_df["Syntyneet"].astype(int)
    birth_death_df["Kuolleet"] = birth_death_df["Kuolleet"].astype(int)

    return birth_death_df

def load_migration_data():
    """
    Lataa muuttoliikedatan täydentävää analysointia varten

    Palauttaa:
        DataFrame: Muuttoliike vuosittain
    """
    migration_path = os.path.join(DATASETS_FOLDER, MIGRATION_FILE)

    # Luetaan Excel
    mig_df = pd.read_excel(migration_path, skiprows=16)
    mig_df = mig_df.dropna()
    mig_df = mig_df.iloc[:24] # Suodatetaan pois epäolennaiset rivit

    # Erotellaan vuodet ja maahan-/maastamuuttodata
    years = mig_df.iloc[:, 0].tolist()
    immigration = mig_df.iloc[:, 4].tolist() # Maahanmuutto yhteensä
    emigration = mig_df.iloc[:, 6].tolist() # Maastamuutto yhteensä

    # Luodaan siistitty DataFrame
    migration_df = pd.DataFrame({"Vuosi": years, "Maahanmuutto": immigration, "Maastamuutto": emigration})

    # Muutetaan DataFrameen mahdolliset desimaaliluvut kokonaisluvuiksi
    migration_df["Vuosi"] = migration_df["Vuosi"].astype(int)
    migration_df["Maahanmuutto"] = migration_df["Maahanmuutto"].astype(int)
    migration_df["Maastamuutto"] = migration_df["Maastamuutto"].astype(int)

    # Lasketaan nettomuuttoliike
    migration_df["Nettomuuttoliike"] = migration_df["Maahanmuutto"] - migration_df["Maastamuutto"]
    migration_df["Nettomuuttoliike"] = migration_df["Nettomuuttoliike"].astype(int)

    return migration_df

def create_features_dataframe():
    """
    Luo yhdistettyjen ominaisuuksien DataFrame mallin opettamista varten

    Palauttaa:
        DataFrame: Yhdistetyt ominaisuudet väkiluvun ennustmaista varten
    """
    print("Ladataan ominaisuus-dataframea...")

    # Ladataan tarvittavat aineistot
    finland_pop_df = load_finland_population_data()
    birth_death_df = load_birth_death_data()
    migration_df = load_migration_data()

    # Yhdistetään aineistojen vuodet
    features_df = finland_pop_df.merge(birth_death_df, on="Vuosi", how="left")
    features_df = features_df.merge(migration_df, on="Vuosi", how="left")

    # Lisätään puuttuvat arvot
    features_df = features_df.interpolate(method="linear")

    # Lasketaan johdetut ominaisuudet (esim. muutokset)
    features_df["Luonnollinen_muutos"] = features_df["Syntyneet"] - features_df["Kuolleet"]
    features_df["Kasvuprosentti"] = features_df["Väkiluku"].pct_change() * 100

    # Lisätään muita ominaisuuksia trendin parempaan havaitsemiseen
    features_df["Väkiluvun_muutos"] = features_df["Väkiluku"].diff()
    features_df["Nettomuutos"] = features_df["Luonnollinen_muutos"] + features_df["Nettomuuttoliike"]

    # Lisätään liukuvia keskiarvoja trendien havaitsemiseen
    features_df["Luonnollinen_muutos_KA3"] = features_df["Luonnollinen_muutos"].rolling(window=3).mean()
    features_df["Nettomuuttoliike_KA3"] = features_df["Nettomuuttoliike"].rolling(window=3).mean()
    features_df["Väkiluvun_muutos_KA3"] = features_df["Väkiluvun_muutos"].rolling(window=3).mean()

    # Eksponentiaaliset liukuvat keskiarvot
    features_df["Väkiluku_EKA3"] = features_df["Väkiluku"].ewm(span=3).mean()

    features_df = features_df.fillna(method="bfill")
    features_df = features_df.fillna(method="ffill")

    return features_df

def prepare_time_series_data(data, target_col, sequnce_length=5):
    """
    Valmistelee aikasarjadatan LSTM-mallia varten.

    Parametrit:
        data: Ominaisuus-DataFrame
        target_col: Ennustettava kohdesarake
        sequence_length: Aika-askeleiden määrä ennustusta varten

    Palauttaa:
        X_train, X_test, y_train, y_test: Opetus- ja testidata
        scaler_X, scaler_y: Ominaisuuksien ja kohteen skaalaaja
    """
    # Skaalataan data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Skaalataan ominaisuudet ja kohde erillään
    data_scaled = scaler_X.fit_transform(data.drop(columns=[target_col]))
    target_scaled = scaler_y.fit_transform(data[[target_col]])

    # Luodaan aikasarja ennustusta varten
    X, y = [], []
    for i in range(len(data) - sequnce_length):
        X.append(data_scaled[i:i+sequnce_length])
        y.append(target_scaled[i+sequnce_length])

    X, y = np.array(X), np.array(y)

    # Jaetaan data opetus- ja testiaineistoihin
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y

def build_lstm_model(input_shape, output_shape=1):
    """
    Rakentaa LSTM-mallin aikasarjaennustusta varten

    Parametrit:
        input_shape: Syötedatan muoto
        output_shape: Ulostulon ominaisuuksien lkm

    Palauttaa:
        model: Rakennettu LSTM-malli
    """
    model = Sequential()

    # Ensimmäinen LSTM-kerros (128 neuronia)
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape, recurrent_regularizer="l2"))

    # Estetään ylisovittuminen pudottamalla satunnaisesti 20% neuroneista
    model.add(Dropout(0.2))

    # Toinen LSTM-kerros (64 neuronia)
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))

    # Kolmas LSTM-kerros (32 neuronia)
    model.add(LSTM(32))
    model.add(Dropout(0.1))

    # Tiheämmät kerrokset tarkempaa ennustusta varten
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(output_shape))

    # Kootaan malli
    model.compile(optimizer=Adam(learning_rate=0.005), loss="mean_squared_error")

    return model

def predict_future_population(model, last_sequence, scaler_X, scaler_y, num_years, feature_cols, features_df):
    """
    Ennustaa väkiluvun tulevaisuudessa opetetun mallin avulla.

    Parametrit:
        model: Opetettu malli
        last_sequence: Viimeinen sarja opetusdatasta
        scaler_X: Ominaisuuksien skaalaaja
        scaler_y: Kohdeskaalaaja
        num_years: Ennustettavat vuodet
        feature_cols: Lista omainaisuuksien sarakkeiden nimistä
        features_df: Alkuperäinen ominaisuuksien DataFrame trendianalyysiin

    Palauttaa:
        list: Ennustettu väkiluvun arvo
    """
    predictions = []
    current_sequence = last_sequence.copy()

    recent_years = 10
    recent_data = features_df.tail(recent_years)

    # Painotetaan enemmän viimeisimpiä vuosia
    weights = np.arange(recent_years, 0, -1)
    weighted_natural_change = np.average(recent_data["Luonnollinen_muutos"], weights=weights)
    weighted_migration = np.average(recent_data["Nettomuuttoliike"], weights=weights)

    for i in range(num_years):
        # Ennustetaan LSTM-mallilla
        pred = model.predict(current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1]))

        # Muunnetaan ennustus takaisin normaaliin muotoon
        pred_orig = scaler_y.inverse_transform(pred)[0][0]

        if i > 0:
            # Hyödynnetään painotettua trendia ensimmäisten ennustusten jälkeen
            last_prediction = predictions[-1]
            trend_based_prediction = last_prediction + weighted_natural_change + weighted_migration

            # Painotetaan mallia ja trendiä
            trend_weight = min(0.4 + (i * 0.1), 0.7)
            model_weight = 1 - trend_weight

            pred_orig = (model_weight * pred_orig) + (trend_weight * trend_based_prediction)

        # Varmistetaan, että ennuste ei ole epärealistinen
        if i > 0:
            max_annual_change = 35000 # Maksimimuutos vuodessa
            prev_pred = predictions[-1]
            change = pred_orig - prev_pred

            if abs(change) > max_annual_change:
                # Rajoitetaan muutosta
                direction = 1 if change > 0 else -1
                pred_orig = prev_pred + (direction * max_annual_change)

        predictions.append(pred_orig)

        # Päivitetään sarja uudella ennustuksella
        new_row = current_sequence[-1].copy()

        # Indeksit tärkeille ominaisuuksille
        try:
            population_idx = feature_cols.index("Väkiluku")
            natural_change_idx = feature_cols.index("Luonnollinen_muutos") if "Luonnollinen_muutos" in feature_cols else None
            migration_idx = feature_cols.index("Nettomuuttoliike") if "Nettomuuttoliike" in feature_cols else None
            growth_rate_idx = feature_cols.index("Kasvuprosentti") if "Kasvuprosentti" in feature_cols else None
            population_change_idx = feature_cols.index("Väkiluvun_muutos") if "Väkiluvun_muutos" in feature_cols else None

            # Päivitetään väkiluku ennustetulla arvolla
            prev_population = scaler_y.inverse_transform([[new_row[population_idx]]])[0][0] if i > 0 else features_df["Väkiluku"].iloc[-1]

            # Skaalataan arvot takaisin normaaliin muotoon ennustusta varten
            new_row[population_idx] = pred[0][0]

            # Päivitetään muut arvot
            if natural_change_idx is not None and i > 0:
                # Arvioidaan luonnollinen muutos viime vuosien trendien perusteella
                natural_change_scaler = MinMaxScaler().fit(recent_data[["Luonnollinen_muutos"]])
                new_row[natural_change_idx] = natural_change_scaler.transform([[weighted_natural_change]])[0][0]

            if migration_idx is not None:
                # Skaalataan arvot
                migration_scaler = MinMaxScaler().fit(recent_data[["Nettomuuttoliike"]])
                new_row[migration_idx] = migration_scaler.transform([[weighted_migration]])[0][0]

            if growth_rate_idx is not None and i > 0:
                # Lasketaan kasvuprosentti
                new_growth_rate = ((pred_orig - prev_population) / prev_population) * 100

                # Skaalataan arvot
                growth_scaler = MinMaxScaler().fit(recent_data[["Kasvuprosentti"]])
                new_row[growth_rate_idx] = growth_scaler.transform([[new_growth_rate]])[0][0]

            if population_change_idx is not None and i > 0:
                # Lasketan absoluuttinen muutos
                pop_change = pred_orig - prev_population

                # Skaalataan arvot
                change_scaler = MinMaxScaler().fit(recent_data[["Väkiluvun_muutos"]])
                new_row[population_change_idx] = change_scaler.transform([[pop_change]])[0][0]

        except (ValueError, IndexError) as e:
            pass

        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = new_row

    return predictions

def predict_city_poplation(city_name, city_df, future_years):
    """
    Ennustaa tietyn kaupungin väkiluvun.

    Parametrit:
        city_name: Kaupunki
        city_df: Kaupungin tietojen DataFrame
        future_years: Ennustettavat vuodet

    Palauttaa:
        dict: Historiallinen ja ennustettu väkiluku vuosittain
    """
    # Erottellaan historiallinen data kaupungille
    historical_data = city_df.loc[city_name].values
    historical_years = city_df.columns.astype(int).tolist()

    # Valmistellaan data sklearn-mallia varten
    X = np.array(historical_years).reshape(-1, 1)
    y = historical_data

    # Luodaan polynomiominaisuudet
    model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    model.fit(X, y)

    # Ennustetaan tulevaisuuden vuodet
    predict_X = np.array(future_years).reshape(-1, 1)
    predictions = model.predict(predict_X)

    # Pyöristetään ennusteet kokonaisluvuiksi
    predictions = np.round(predictions).astype(int)

    # Yhdistetään historiallinen ja ennustettu data
    result = {}
    for year, population in zip(historical_years, historical_data):
        result[year] = int(population)

    for year, population in zip(future_years, predictions):
        result[year] = int(population)

    return result

def predict_finland_population(features_df, future_years):
    """
    Ennustaa Suomen väkiluvun LSTM-mallin avulla

    Parametrit:
        features_df: Omainaisuuksien DataFrame
        future_years: Ennustettavat vuodet

    Palauttaa:
        dict: Historiallinen ja ennustettu väkiluku vuosittain
    """
    # Määritellään ominaisuuksien sarakkeet
    feature_cols = ["Väkiluku", "Syntyneet", "Kuolleet", "Nettomuuttoliike", 
                    "Luonnollinen_muutos", "Kasvuprosentti", "Väkiluvun_muutos",
                    "Nettomuutos", "Luonnollinen_muutos_KA3", "Nettomuuttoliike_KA3",
                    "Väkiluku_muutos_KA3", "Väkiluku_EKA3"]
    
    available_cols = [col for col in feature_cols if col in features_df.columns]

    # Valmistellaan aikasarjadata
    sequence_length = 10
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = prepare_time_series_data(
        features_df[available_cols], "Väkiluku", sequence_length
    )

    # Rakennetaan ja opetetaan LSTM-malli
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)

    # Estetään ylisovittuminen
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    # Opetetaan malli
    history = model.fit(
        X_train, y_train, 
        epochs=100, 
        batch_size=8, 
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=0
        )

    # Haetaan ennustamista varten viimeinen sarja
    last_sequence = X_test[-1]

    # Ennustetaaan tulevaisuuden väkiluku
    predictions = predict_future_population(
        model, last_sequence, scaler_X, scaler_y,
        len(future_years), available_cols, features_df
    )
    
    # Yhdistetään historiallinen ja ennustettu data
    result = {}
    historical_years = features_df["Vuosi"].values
    historical_population = features_df["Väkiluku"].values

    for year, population in zip(historical_years, historical_population):
        result[year] = int(population)

    for year, population in zip(future_years, predictions):
        result[year] = int(population)

    return result

def plot_population(result, title, current_year=2023):
    """
    Piirtää historiallisen ja ennustetun väkiluvun.

    Parametrit:
        result: Sanakirja, jossa vuodet avaimina ja väkiluku arvoina
        title: Otsikko
        current_year: Vuosi, joka erottaa historiallisen datan ennustuksista
    """
    years = sorted(result.keys())
    population = [result[year] for year in years]

    # Jaetaan historialliseen ja ennustettuun
    historical_years = [year for year in years if year <= current_year]
    historical_population = [result[year] for year in historical_years]

    predicted_years = [year for year in years if year > current_year]
    predicted_population = [result[year] for year in predicted_years]

    plt.figure(figsize=(12, 6))

    # Piirretään historiallinen data
    plt.plot(historical_years, historical_population, "bo-", label="Historiallinen data")

    # Piirretään ennustettu data
    plt.plot(predicted_years, predicted_population, "ro--", label="Ennustettu data")

    # Piirretään pystyviiva kohtaan, jossa ennustus alkaa
    plt.axvline(x=current_year, color="gray", linestyle="--", alpha=0.7, label="Ennustuksen alku")

    # Yhdistetään viimeinen vuosi ja ensimmäinen ennustettu vuosi viivalla
    if predicted_years and historical_years:
        connecting_x = [historical_years[-1], predicted_years[0]]
        connecting_y = [historical_population[-1], predicted_population[0]]
        plt.plot(connecting_x, connecting_y, "o-", color="orange")

    y_min = min(population) * 0.9
    y_max = max(population) * 1.1
    plt.ylim(y_min, y_max)

    plt.title(title)
    plt.xlabel("Vuosi")
    plt.ylabel("Väkiluku")
    plt.legend()
    plt.grid(True, alpha=0.3)

    step = max(1, len(years) // 10)
    plt.xticks(years[::step])

    plt.tight_layout()
    plt.show()    

def main():
    # Ladataan kaupunkidata
    city_df = load_city_data()

    # Haetaan suurimmat (top 10) kaupungit väkiluvultaan
    most_recent_year = city_df.columns[-1]
    top_cities = city_df[most_recent_year].sort_values(ascending=False).head(10).index.tolist()

    # Luodaan Suomen ennustamista varten ominaisuuksien DataFrame
    features_df = create_features_dataframe()

    # Ennustettavat vuodet (2024-2040)
    future_years = list(range(2024, 2041))

    while True:
        print("\nValitse vaihtoehto:")
        print("0. Ennusta Suomen väkiluku")
        for i, city in enumerate(top_cities, 1):
            print(f"{i}. Ennusta kaupungin {city} väkiluku")

        try:
            choice = input("\nSyötä luku (0-10) tai paina Enter jatkaaksesi: ")

            if choice == "":
                # Kysytään tarkempaa vuotta
                year_input = input("Syötä vuosi (2010-2040): ")
                year = int(year_input)

                if 2010 <= year <= 2040:
                    print("\nMitä ennustetaan ?")
                    print("0. Suomen väkiluku")
                    for i, city in enumerate(top_cities, 1):
                        print(f"{i}. Kaupungin {city} väkiluku")

                    entity_choice = input("Syötä luku (0-10): ")
                    entity_choice = int(entity_choice)

                    if 0 <= entity_choice <= 10:
                        if entity_choice == 0:
                            # Ennustetaan suomen väkiluku
                            entity_name = "Suomi"
                            if year <= 2023:
                                # Historiallinen data
                                year_data = features_df[features_df["Vuosi"] == year]
                                if not year_data.empty:
                                    population = int(year_data["Väkiluku"].values[0])
                                    print(f"\n{entity_name}: Väkiluku vuonna {year} oli {population} ihmistä.")
                                else:
                                    print(f"{entity_name}: Ei dataa vuodelle {year}.")
                            else:
                                # Ennustus
                                result = predict_finland_population(features_df, future_years)
                                if year in result:
                                    population = result[year]
                                    print(f"{entity_name}: Väkiluku vuonna {year} on {population} ihmistä.")
                                else:
                                    print(f"Ei voitu ennustaa väkilukua vuodelle {year}.")
                        else:
                            # Ennustetaan kaupungin väkiluku
                            city_name = top_cities[entity_choice - 1]
                            if year <= 2023 and year in city_df.columns:
                                # Historiallinen data
                                population = int(city_df.loc[city_name, year])
                                print(f"\n{city_name}: väkiluku vuonna {year} oli {population} ihmistä.")
                            else:
                                # Ennustus
                                result = predict_city_poplation(city_name, city_df, future_years)
                                if year in result:
                                    population = result[year]
                                    print(f"\n{city_name}: Väkiluku vuonna {year} on {population} ihmistä.")
                                else:
                                    print(f"Ei voitu ennustaa väkilukua vuodelle {year}.")
                    else:
                        print("Epäkelpo vaihoehto. Valitse numero väliltä 0-10.")
                else:
                    print("Epäkelpo vuosi. Valitse vuosi väliltä 2010-2040.")

            elif choice.isdigit():
                choice = int(choice)
                if 0 <= choice <= 10:
                    # Haetaan ennustettavat vuodet visualisointia varten
                    prediction_years = list(range(2010, 2041))

                    if choice == 0:
                        # Ennustetaan Suomen väkiluku
                        result = predict_finland_population(features_df, future_years)

                        # Suodetaan ennusteen vuodet (2010 eteenpäin)
                        result = {year: pop for year, pop in result.items() if year >= 2010}

                        # Näytetään tulokset
                        print("\nSuomen väkiluvun ennuste:")
                        for year in sorted(result.keys()):
                            if year <= 2023:
                                print(f"{year}: {result[year]}")
                            else:
                                print(f"{year}: {result[year]} (ennustettu)")

                        # Piirretään tulokset
                        plot_population(result, "Suomen väkiluvun ennuste")
                    else:
                        # Ennustetaan kaupungin väkiluku
                        city_name = top_cities[choice - 1]

                        # Haetaan historiallinen data (2010 eteenpäin)
                        historical_years = [year for year in city_df.columns if year >= 2010 and year <= 2023]

                        # Ennustus
                        result = predict_city_poplation(city_name, city_df, future_years)

                        # Näytetään tulokset
                        print(f"\nKaupungin {city_name} väkiluvun ennuste:")
                        for year in sorted(result.keys()):
                            if year <= 2023:
                                print(f"{year}: {result[year]}")
                            else:
                                print(f"{year}: {result[year]} (ennustettu)")

                        # Piirretään tulokset
                        plot_population(result, f"Kaupungin {city_name} väkiluvun ennuste")
                else:
                    print("Epäkelpo vaihtoehto. Valitse numero väliltä 0-10")
            else:
                print("Epäkelpo syöte. Valitse numero tai paina Enter")

        except ValueError as e:
            print(f"Virhe: {e}. Yritä uudelleen.")

        continue_choice = input("\nHaluatko tehdä toisen ennustuksen? (k/e): ")
        if continue_choice.lower() != 'k':
            break

if __name__ == "__main__":
    main()