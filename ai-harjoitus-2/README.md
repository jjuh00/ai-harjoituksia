## Hotelliarvostelun analysointiohjelma

Ohjelma, joka analysoi TripAdvisorin hotelliarvosteluja luonnollisen kielen käsittelyn ja koneoppimisen
menetelmien avulla. Tarkoituksena oli harjoitella/hyödyntää opittuja taitoja tekoälyn ohjelmoinin kurssilta.
Ohjelma on testattu toimivaksi Python 3.12 -versiolla.

## Kuvaus

Ohjelma lataa hotelliarvosteluja sisältävän aineiston ja suorittaa sille monivaiheisen analyysin:
  - Tekstien esikäsittely (puhdistaminen, normalisointi)
  - Tunneanalyysi (positiivinen, neutraali, negativiinen)
  - Avainsanojen erittely luonnollisen kielen käsittelyn avulla
  - TF-IDF-vektorointi ja KMeans-klusterointi
  - Visualisoinnit tuloksita (mm. avainsanat ja tunteet)

## Aineisto
Ohjelma hyödyntää TridAdvisor-hotelliarvosteluaineistoa. Aineiston lähde: [Trip Advisor Hotel Reviews](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews)

## Ohjelman ajaminen
Seuraavat komennot ajetaan hakemistossa ai-harjoituksia/ai-harjoitus-2.<br />
Asenna tarvittavat kirjastot seuraavalla komennolla: pip install pandas nltk textblob scikit-learn numpy matplotlib seaborn

Ohjelma ajetaan komennolla python main.py

## Mallin yksityiskohdat
- **Tunneanalyysi**: TextBlob-tunneanalyysi, joka määrittelee tekstin polariteetin.
- **Avainsanojen tunnistus**: NLTK:n POS-merkintöjen avulla valitaan substantiivit ja adjektiivit
- **Klusterointi**: TF-IDF-vektorointi yhdistetty KMeans-algoritmiin
