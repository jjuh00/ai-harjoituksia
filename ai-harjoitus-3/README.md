## Opiskelijoiden koesuoritusten ennustus- ja vinouma-analyysiohjelma

Ohjelma, joka ennustaa opiskelijan kokeen läpäisyn väestö- ja opiskelutietojen perusteella sekä 
analysoi mahdollisia vinoumia eri ihmisryhmien välillä. Tarkoituksena oli harjoitella/hyödyntää opittuja taitoja tekoälyn ohjelmoinin kurssilta ja
vastuullisen tekoälyn soveltamista käytännössä.<br />
Ohjelma on testattu toimivaksi Python 3.12 -versiolla.

## Kuvaus

Ohjelman kulku:
  - Datan esikäsittely ja kategoristen muuttujien enkoodaus
  - Binäärisen kohdemuuttujan luonti (kokeen läpäisy)
  - Koneoppimismallien opettaminen ja arviointi (logistinen regressio, SVM, satunnaismetsä, Naiivi Bayes, päätöspuu)
  - Mallien suorituskyvyn vertailu (tarkkuus, herkkyys, F1)
  - Visualisoinnit alkupäerisestä datasta ja mallien arvioinneista
  - Vinouma-analyysi Fairlearn-kirjaston avulla

## Aineisto
[Students Performace in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

## Ohjelman ajaminen

Seuraavat komennot ajetaan hakemistossa ai-harjoituksia/ai-harjoitus-3.<br />
Asenna tarvittavat kirjastot seuraavalla komennolla: pip install pandas scikit-learn matplotlib numpy seaborn fairlearn

Ohjelma ajetaan komennolla python main.py

## Ohjelman yksityiskohdat

- Esikäsittely: Kategoriset muuttujan enkoodataan ja pisteistä lasketaan keskiarvo
- Kohdemuuttuja: Opiskelija läpäisee kokeen, jos kolmen osa-alueen keskiarvo on >= 50
- Koneoppimismallien vertailu: Mittareina tarkkuus, sisäinen tarkkuus, herkkyys ja F1-arvo
- Vinouma-analyysi:
  - Sukupuolen ja etnisen tausta mukaan
  - Mittareina väestöllisen tasa-arvon ero (demographic parity difference) ja tasavertaisen kertoiminen ero
    (equalized odds difference)
  - Visualisonti pylväsdiagrammeina
