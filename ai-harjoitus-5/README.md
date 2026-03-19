## Telco-asiakaspysyvyysdatan monitorointiohjelma

Ohjelma, joka ennustaa asiakkaiden poistumsta, seuraa mallin suorituskykyä ajan kuluessa, havaitsee datan ajautumisen (drift) ja tarjoaa visuaalisen hallintapaneelin analysointiin. <br />

## Kuvaus

Ohjelman kulku:
  - Datan esikäsittely, kategoristen muuttujien enkoodaus ja kohdemuuttujan muuttaminen binääriskesi
  - Koneoppimismallin koulutus ja arviointi
  - Mallin, tilastojen ja ennusteiden tallennus
  - Monitoroinnissa lasketaan [PSI](https://www.geeksforgeeks.org/data-science/population-stability-index-psi/), [KS-testi](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test) ja suorituskykymittarit vertailemalla referensseihin
  - Streamlit-hallintapaneeli visualisoi datajakaumat, ajauma-analyysin, suorituskykymittarit ja hälytykset

## Aineisto
IBM Sample Datasets. Telco Customer Churn dataset. Saatavilla: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Ohjelman ajaminen

Asenna tarvittaessa Python 3.14. Kaikki komennot annetaan hakemistossa ai-harjoituksia/ai-harjoitus-5. <br />
1. Asenna kirjastot
```bash
pip install -r requirements.txt
```
2. Kouluta malli
```bash
python src/train.py
```
3. Suorita monitoroinit
```bash
python src/monitor.py
```
4. Käynnistä Streamlit
```bash
python -m streamlit run src/dashboard.py
```
5. (Vaihtoehto) Aja kaikki
```bash
python run_all.py
```
6. (Vaihtoehto) Suorita monitorointi ja käynnistä Streamlit jo koulutetulle sekä tallennetulle mallille
```bash
python run_all.py --skip-train
```
7. (Vaihtoehto) Suorita mallin koulutus ja monitorointi, mutta älä käynnistä Streamlit:iä
```bash
python run_all.py --no-dashboard
```

## Ohjelman yksityiskohdat

- Samaa aineistoa käytetään myös mallin arviointiin sekä seurantaan, mutta niihin käytetään satunnaisnäytteitä
- Esikäsittely: Skaalaus StandardScalerilla, enkoodaus One-Hot Encodingilla
- Malli: GradientBoostingClassifier
- Mallin arviointi: Tarkkuus, sisäinen tarkkuus, herkkys, f1-pisteet ja ROC-AUC
- Monitorointi: Data-ajautuminen
  - PSI
    - < 0.10 -> OK
    - 0.10-0.20 -> varoitus
    - 0.20 -> hälytys
  - KS-testi
    - p < 0.05 -> merkittävä muutos
-Suorituskykymittariseuranta toteutetaan vertailulla referenssimittareihin. Hälytys tapahtuu, jos lasku on suurempi kuin 5% verrattuna referenssiin
- Hallintapaneeli
  - Histogrammit (referenssi vs. nykyinen)
  - PSI-visualisointi
  - KS-testit
  - Sekaannusmatriisi
  - Hälytykset
