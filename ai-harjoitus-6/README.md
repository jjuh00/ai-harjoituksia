# A* ja Dijkstra -reitinhakija

Ohjelma vertailee kahta reitinhakualgoritmia, Dijkstraa ja A*:a, samalla ruudukolla tai painotetulla verkolla. Dijkstra etsii lyhimmän polun laajenemalla tasaisesti joka suuntaan alkupisteestä. A* käyttää lisäksi heuristiikkaa h(s), joka arvioi jäljellä olevan matkan päätepisteeseen, ja ohjaa hakua kohti maalia; löytäen saman optimaalisen polun, mutta yleensä tutkien vähemmän solmuja, kunhan heuristiikka on kelvollinen (ei koskaan yliarvioi todellista kustannusta). Molemmat algoritmit ovat itse asiassa saman yleisen haun (`best_first_search`) erikoistapauksia: Dijkstra on A*, jonka heuristiikka on nolla.

## Kuvaus

Ohjelma lukee kartan (se on joko tiedostossa (.txt) tai se generoidaan satunnaisesti komennolla) tai CSV-verkon, etsii lähtö- ja päätepisteen, ja ajaa sekä Dijkstran että valitut A*-heuristiikat samalla kartalla. Jokaisesta ajosta mitataan polun kustannus, laajennettujen/vierailtujen solmujen määrä ja ajoaika.

## Ohjelman ajaminen

Asenna tarvittaessa Python (toimii uusimmilla versioilla). Halutessasi asenna matplotlib komennolla `pip install matplotlib`. Kaikki alla olevat komennot annetaan hakemistossa ai-harjoituksia/ai-harjoitus-6. <br />

### Esimerkkikomentoja

Ruudukkotila tiedostosta (voi olla myös large.txt ja terrain.txt)
```bash
python src/main.py --ruudukko ./maps/maze.txt --diagonaali
```

Satunnainen ruudukko:
```bash
python src/main.py --satunnainen 30 30 --siemenluku 42 --heuristiikat euklidinen manhattan
```

CSV-verkkotila:
```bash
python src/main.py --solmut ./data/nodes.csv --karet ./data/edges.csv --lahto A --maali C
```

Kaikki parametrit ja niiden selitykset:
```bash
python src/main.py --help
```

### Komentoriviparametrit

Lisäparametrit: `--diagonaali` (8-suuntainen liike), `--heuristiikat` (mikä/mitkä A*-heuristiikat ajetaan), `--kaksisuuntainen` (kaksisuuntainen Dijkstra), `--nayta-ruudukko` (tulostaa alkuperäisen kartan ennen reitin etsintää), `--tallenna-kuva polku/tiedosto.png` (vaatii matplotlibin, tallentaa kuvan kartasta ja polusta)

## Yksityiskohdat

- **Ruudukon merkit**: `.` vapaa (paino 1.0), `#` seinä, `A` alkupiste, `L` päätepiste, `^` raskas maasto (paino 3.0), `,` kevyt maasto (paino 1.5)
- **Naapurit**: 4- tai 8-suuntainen liike; diagonaalit maksavat √2 ja nurkkien "läpileikkaus" seinien välistä estetään oletuksena.
- **Heuristiikat**: Manhattan (epäpätevä 8-suuntaisessa liikkeessä), Euklidinen (kelvollinen molemmissa, mutta löysä), Chebyshev (löydä vertailukohta), Octile (tiukka ja suositeltu 8-suuntaiseen liikkeeseen, kun diagonaali maksaa √2).
- **Prioriteettijono**: Koska Pythonin heapq ei tue "decrease-key"-operaaatiota, parempi reitti lisätään jonoon uutena merkintänä ja vanhentuneet merkinnät ohitetaan laiskasti postettaessa.
- **CSV-verkkotila**: Yleistää haun mielivaltaisiin painotettuihin verkkoihin (esim. tieverkko); A* käyttää solmujen koordinaattien euklidista etäisyyttä heuristiikkana.
- **Kaksisuuntainen haku**: Ajaa hakua samanaikaisesti molemmista päistä ja pysähtyy, kun hakusuunnat kohtaavat.