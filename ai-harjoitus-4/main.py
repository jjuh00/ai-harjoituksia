"""
RAG-kysymys-vastausjärjestelmä

Tämä on hakutoimintoa tukeva generointijärjestelmä (RAG), jonka avulla käyttäjät voivat:
1. Ladata ja käsitellä dokumentteja (PDF, TXT)
2. Esittää kysymyksiä dokumenttien sisällöstä
3. Saada vastauksia, jotka perustuvat ladattuihin dokumentteihin

Järjestelmä käyttää:
-sentence-transformers-mallia semanttisten upotusten luomiseen
-Vektorin kosinietäisyyttä relevanttien dokumenttien osien hakemiseen
-HuggingFace Transformers -mallia vastausten luomiseen
-PySide6-kirjastoa graafisen käyttöliittymän rakentamiseen
"""

from PySide6.QtWidgets import QApplication
import sys
from ui.main_window import MainWindow

def main():
    """Alustaa Qt-sovelluksen ja näyttää pääikkunan."""
    # Luodaan Qt-sovellus
    app = QApplication(sys.argv)

    app.setApplicationName("RAG-järjestelmä")
    app.setApplicationVersion("1.0.0")

    # Luodaan ja näytetään pääikkuna
    window = MainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()