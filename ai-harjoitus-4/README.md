# RAG-järjestelmä

RAG-järjestelmä (Retrieval-Augmented Generation) on hakutoimitoon perustuva kysymys-vastausjärjestelmä,
joka mahdollistaa käyttäjien ladata omia dokumenttejaan ja esittää niistä kysymyksiä luonnollisella kielellä.
Järjestelmä hyödyntää semanttista hakua ja kielimalleja tuottaakseen vastauksia dokumenttien sisällöstä.
Testattu Python 3.14 -versiolla.

## Kuvaus

Ohjelman kulku:
 - Käyttäjän lataama dokumentti (.txt tai .pdf) pilkotaan osiin (chunks).
 - Dokumentin osille luodaan vektoriupotukset (embedding).
 - Käyttäjä kysyy kysymyksen dokumentista, jolloin kysymyksellekin luodaan vektoriupotus. Järjestelmä
   laskee kosinietäisyyden kyselyn ja kaikkien dokumenttiosien välillä.
 - Relevanteimmat osat haetaan ja järjestetään samankaltaisuuspisteiden mukaan.
 - Kielimalli generoi vastauksen perustuen sekä kysymykseen että haettun kontekstiin.

## Ohjelman ajaminen
Siirry hakemistoon ai-harjoitus-4 cd-komennolla.<br />
Tarvittaessa asenna Python 3.14 ja pip.
Asenna tarvittavat kirjastot komennolla ```pip install -r requirements.txt```.
Ohjelma käynnistetään komennolla ```python main.py```.

## Ohjelman yksityiskohdat

- Dokumenttien pilkkominen: Dokumentit pilkotaan oletusarvoisesti 512-merkin osiin. Jokainen osa sisältää viittauksen alkuperäiseen dokumenttiin.
- Vektoriupotukset: Mallina käytetään esikoulutettua sentence-transformers-mallia, jonka upotusulottuvuus on 384. Se käyttää L2-normia kosinietäisyyttä ja normalisointia varten.
- Haku: Samankaltaisuus mittana käytetään kosinietäisyyttä. Tehokkuuden lisäämiseksi käytetään NumPy-vektorisoituja operaatioita.
- Kielimalli: Vastauksen luomiseen käytetään google/flan-t5-base-mallia, jonka vastauksen pituus on max. 512 tokenia.
- Käyttöliittymä: PySide6 (Qt for Python), taustaprosesseita varten ohjelma käyttää QThreade-pohjaisia työskentelysäikeitä.

## Rajoitukset ja huomiot

 - Kysymys tulee esittää englanniksi, sillä kielimalli tukee englantia.
 - Jos käytettävällä laiteella on Nvidia-näytönohjain, kannattaa asentaa CUDA, jonka avulla ohjelma on huomattavasti suorituskykyisempi.
 - PDF-dokumenttien tulee olla tekstipohjaisia (eli ne eivät saa sisältää kuvia).
 - Ensimmäinen käynnistys voi kestää jonkin aikaa, kun mallit ladataan HuggingFacesta.
