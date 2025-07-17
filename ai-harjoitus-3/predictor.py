"""
Opiskelijan koesuorituksen ennustus- ja vinouman analysointiohjelma.

Ohjelma käyttää koneoppimista ennustamaan opiskelijan koesuorituksen tuloksia väestörakenteen ja
opintoihin liittyvien tekijöiden perusteella. Se käyttää useita luokittelumenetelmiä ennustamaan,
läpäisevätkö opiskelijat kokeen vai eivät.

Ohjelma esikäsittelee datan, enkoodaa kategoriset ominaisuudet, opettaa useita luokittelumalleja ja arvioi niiden suorituskyvyn.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATASET_FILE = "StudentsPerformance.csv"

class DataPreprocessor:
    """
    Käsittelee datan lataamisen, esikäsittelyn ja ominaisuuksien järjestelyn opiskelijan koesuorituksen ennustamista varten.

    Raaka data muutetaan sellaiseen muotoon, että sitä voidaan käyttää koneoppimismallien opettamiseen, 
    kategoristen ominaisuuksien enkoodaukseen ja kohdemuuttujan erottelemiseen.
    """
    def __init__(self):
        self.data = None
        self.label_encoders = {}
        self.sensitive_features = ["gender", "race/ethnicity"]

    def load_data(self, dataset_file):
        """
        Lataa datan CSV-tiedostosta.
        
        Parametrit:
            dataset_file (str): CSV-tiedoston nimi
            
        Palauttaa:
            pd.DataFrame: Ladattu data
        """
        try:
            self.data = pd.read_csv(dataset_file)
            print(f"Aineisto ladattu onnistuneesti. Aineiston koko: {self.data.shape}")
            return self.data
        except FileNotFoundError:
            return None
        except Exception:
            return None
        
    def create_target_variable(self):
        """
        Luo binäärisen kohdemuuttujan opiskelijan saamien pisteiden
        keskiarvon perusteella.
        
        Opiskelija pääsee läpi kokeen, jos pisteiden (matikka+lukutaito+kirjoitustaito / 3) keskiarvo on vähintään 50.

        Palauttaa:
            pd.Series: Kohdemuuttuja, jossa 1 = läpäisi kokeen, 0 = ei läpäissyt
        """
        if self.data is None:
            self.load_data(DATASET_FILE)
        
        # Lasketaan keskiarvo pisteistä
        avg_score = (self.data["math score"] +
                     self.data["reading score"] +
                     self.data["writing score"]) / 3
        
        # Luodaan binäärinen kohdemuuttuja (1 = läpäisi, 0 = ei läpäissyt)
        self.data["pass"] = (avg_score >= 50).astype(int)

        return self.data["pass"]
    
    def encode_categorical_features(self):
        """
        Enkoodaa kategoriset ominaisuudet LabelEncoderilla.

        Palauttaa:
            pd.DataFrame: Data, jossa kategoriset ominaisuudet on enkoodattu
        """
        if self.data is None:
            self.load_data(DATASET_FILE)

        categorical_columns = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]

        # Luodaan kopio alkuperäisestä datasta, jotta alkuperäinen data säilyy muuttumattomana
        encoded_data = self.data.copy()

        for column in categorical_columns:
            if column in encoded_data.columns:
                le = LabelEncoder()
                encoded_data[column] = le.fit_transform(encoded_data[column])
                self.label_encoders[column] = le
                # print(column, self.label_encoders[column].classes_)

        return encoded_data

    def get_sensitive_features(self, encoded_data):
        """
        Erottaa herkät ominaisuudet datasta vinouman analysointiin.

        Parametrit:
            encoded_data (pd.DataFrame): Enkoodattu data, josta herkät ominaisuudet halutaan erottaa

        Palauttaa:
            pd.DataFrame: Data, jossa on vain herkät ominaisuudet
        """
        return encoded_data[self.sensitive_features].copy()
    
    def prepare_features_and_target(self, encoded_data):
        """
        Valmistaa ominaisuudet ja kohdemuuttujan mallin opettamista varten.

        Parametrit:
            encoded_data (pd.DataFrame): Enkoodattu data, jossa on kategoriset ominaisuudet

        Palauttaa:
            tuple: (X, y), missä X on ominaisuusmatriisi ja y on kohdemuuuttuja
        """
        # Valitaan ominaisuussarakkeet (ei otetaan huomioon kohdetta ja yksittäisiä pisteitä)
        feature_columns = ["gender", "race/ethnicity", "parental level of education", "lunch",
                           "test preparation course", "math score", "reading score", "writing score"]
        
        X = encoded_data[feature_columns]
        y = encoded_data["pass"]

        return X, y
    
class ModelTrainer:
    """
    Käsitellee useiden mallien opettamisen ja niieden arvioinnin.

    Ohjelma opettaa useita luokittelumalleja opiskelijan koesuorituksen ennustamiseen ja 
    arvioi niiden suorituskyvyn vertailua ja vinouman analysointia varten.
    """
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.scaler = StandardScaler()
        self.X_train_scaled = None
        self.X_test_scaled = None

    def _initialize_models(self):
        """
        Alustaa eri luokittelumallit.
        """
        self.models = {
            "Logistinen regressio": LogisticRegression(max_iter=1000, random_state=42),
            "Tukivektorikone (SVM)": SVC(probability=True, random_state=42),
            "Satunnaismetsä": RandomForestClassifier(n_estimators=100, random_state=42),
            "Naiivi Bayes": GaussianNB(),
            "Päätöspuu": DecisionTreeClassifier(max_depth=10, random_state=42) # max_depth ylisovittumisen estämiseksi
        }

    def prepare_data(self, X, y):
        """
        Jakaa datan opetus- ja testiaineistoihin ja skaalaa ominaisuudet.
        
        Parametrit:
            X (pd.DataFrame): Ominaisuusmatriisi
            y (pd.Series): Kohdemuuttuja
            
        Palauttaa:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Jaetaan data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Skaalataan ominaisuudet
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """
        Opettaa kaikki alustetut mallit opetusdatalla.
        
        Parametrit:
            X_train (pd.DataFrame): Ominaisuusmatriisi opetukseen
            y_train (pd.Series): Opetuskohdemuuttuja
            
        Palauttaa:
            dict: Sanakirja opetetuist malleista
        """
        if not self.models:
            self._initialize_models()

        for name, model in self.models.items():
            # Käytetään skaalattua dataa malleille, jotka hyötyvät skaalatusta datasta
            if name in ["Logistinen regressio", "Tukivektorikone (SVM)"]:
                model.fit(self.X_train_scaled, y_train)
            else:
                model.fit(X_train, y_train)

            self.trained_models[name] = model

        return self.trained_models
    
    def evaluate_models(self, X_test, y_test):
        """
        Arvioi opetettujen mallien suorituskyvyn.
        
        Parametrit:
            X_test (pd.DataFrame): Ominaisuusmatriisi testaukseen
            y_test (pd.Series): Kohdemuuttuja
            
        Palauttaa:
            dict: Sanakirja mallien arviointituloksista
        """
        results = {}

        for name, model in self.trained_models.items():
            # Käytetäään skaalattua dataa malleille, jotka opetettiin skaalatulla datalla
            if name in ["Logistinen regressio", "Tukivektorikone (SVM)"]:
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            else:
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Lasketaan arvioinnit
            results[name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
                "predictions": y_pred,
                "prediction_probabilities": y_pred_proba
            }

        return results
    
class StudentPerformancePredictor:
    """
    Opiskelijan koesuorituksen ennustaja, joka yhdistää datan esikäsittelyn ja mallin opettamisen.
    """
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.trainer = ModelTrainer()
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.sensitive_features = None
        self.results = None

    def load_and_prepare_data(self):
        """
        Lataa ja esikäsittelee datan mallien opettamiseen.

        Palauttaa:
            bool: True, jos datan esivalmistelu onnistui, muuten False
        """
        # Ladataan data
        self.data = self.preprocessor.load_data(DATASET_FILE)
        if self.data is None:
            return False
        
        # Luodaan kohdemuuttuja
        self.preprocessor.create_target_variable()

        # Enkoodataan kategoriset ominaisuudet
        encoded_data = self.preprocessor.encode_categorical_features()

        # Erotellaan herkät ominaisuudet
        self.sensitive_features = self.preprocessor.get_sensitive_features(encoded_data)

        # Valmistellaan ominaisuudet ja kohdemuuttuja
        self.X, self.y = self.preprocessor.prepare_features_and_target(encoded_data)

        return True

    def train_and_evaluate_models(self):
        """
        Opettaa ja arvioi kaikki mallit.
        
        Palauttaa:
            dict: Mallien arviointitulokset
        """
        # Valmistellaan data opetukseen
        self.X_train, self.X_test, self.y_train, self.y_test = self.trainer.prepare_data(self.X, self.y)

        # Opetetaan mallit
        self.trainer.train_models(self.X_train, self.y_train)

        # Arvioidaan mallit
        self.results = self.trainer.evaluate_models(self.X_test, self.y_test)

        return self.results
    
    def get_model_performances(self):
        """
        Hakee mallien suorituskyvyn tulokset.

        Palauttaa:
            pd.DataFrame: Mallien arviointitulokset
        """
        if self.results is None:
            self.train_and_evaluate_models()

        summary = []
        for model_name, metrics in self.results.items():
            summary.append({
                "Malli": model_name,
                "Tarkkuus": metrics["accuracy"],
                "Tarkkuus (sisäinen)": metrics["precision"],
                "Herkkyys": metrics["recall"],
                "F1-arvo": metrics["f1_score"]
            })

        return pd.DataFrame(summary)
    
    def get_model_predictions_for_bias_analysis(self, model_name):
        """
        Hakee ennustukset ja herkät ominaisuudet vinouman analysointia varten.

        Parametrit:
            model_name (str): Mallin nimi, jolle ennustukset halutaan

        Palauttaa:
            tuple: (predictions, sensitive_features_test, y_test)
        """
        if self.results is None:
            self.train_and_evaluate_models()

        # Haetaan ennustukset tuloksista
        predictions = self.results[model_name]["predictions"]

        # Haetaan vastaavat herkät ominaisuudet testidatasta
        sensitive_features_test = self.sensitive_features.iloc[self.X_test.index]

        return predictions, sensitive_features_test, self.y_test