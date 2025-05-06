## Suomen väestön ennustusohjelma

Koneoppimista hyödyntävä ohjelma, joka ennustaa Suomen ja suurimpien kaupunkien väestötrendin vuosina 2010-2040. Tarkoituksena oli harjoitella/hyödyntää opittuja taitoja tekoälyn ohjelmoinnin kurssilta. Ohjelma on testattu toimivaksi Python 3.12 -versiolla.

## Kuvaus

Ohjelma, joka hyödyntää historiallisia väestötietoja ennustemallien rakentamiseen Suomen ja sen suurimien kaupunkien väestökasvun ennustamiseksi. Ohjelma yhdistää erilaisia aineistoja, jotka sisältävät väestötilastoja, syntyvyys- ja kuolleisuuslukuja sekä muuttoliikemalleja.<br/>
Omaisuudet:
  - Tietojen esikäsittely Excel-tiedostoista
  - Neuroverkkomalli (LSTM) Suomen väestön ennustamiselle
  - Polynomiregressio kaupunkikohtaisia ennusteita varten
  - Visualisoinnit historiatiedoista ja tulevaisuuden ennusteista

## Aineistot
Ohjelma hyödyntää neljää aineistoa Suomen väestörakenteesta:
  1. Suurimpien kuntie väkiluku (1980-2023)
  2. Väestö sukupuolen ja iän mukaan (1870-2075)
  3. Elävänä syntyneet ja kuolleet (1750-2023)
  4. Muuttoliike (2000-2023)

Aineistot ovat Tilastokeskuksen kehittämiä. Aineistojen lähde: [Tilastokeskus Väestö ja yhteiskunta] (https://pxhopea2.stat.fi/sahkoiset_julkaisut/vuosikirja2024/html/suom0011.htm)

## Ohjelman ajaminen

Asenna tarvittavat kirjastot seuraavalla komennolla: pip install pandas openpyxl numpy matplotlib scikit-learn tensorflow

Ohjelma ajetaan komennolla python main.py

## Mallin yksityiskohdat
 - **Suomen väestö**: LSTM-neuroverkko, joka hyödyntää useita omiansuuksia (syntyneet, kuolleet, muuttoliike jne.) väestörakenteiden ennustamiseen.
 - **Kaupunkien väestö**: Polynomiregressio, joka kuvaa epälineaariisia kasvu- ja laskusuuntauksia.
