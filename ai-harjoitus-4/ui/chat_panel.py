"""
Chat-käyttöliittymäkomponentti kysymysten esittämiseen ja vastausten näyttämiseen.
Sisältää myös keskusteluhistorian ja kontekstin visualisoinnin.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QScrollArea,
    QSplitter, QLineEdit, QPushButton
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QTextCursor

class ContextChunkWidget(QWidget):
    """Widget yksittäisen kontekstiosan näyttämiseen."""

    def __init__(self, chunk_info, parent=None):
        """
        Parametrit ja attribuutit:
            chunk_info (dict): Sanakirja, joka sisältää kontekstiosan tiedot.
            parent (Any): Vanhempi (parent) widget.
        """
        super().__init__(parent)
        self.chunk_info = chunk_info
        self._setup_ui()

    def _setup_ui(self):
        """Asettaa käyttöliittymän komponentit."""
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # Header ja pisteet
        header_layout = QHBoxLayout()

        filename_label = QLabel(f"{self.chunk_info['doc_filename']}")
        filename_label.setStyleSheet("font-weight: bold;")

        score = self.chunk_info["similarity_score"]
        score_label = QLabel(f"Pisteet: {score:.3f}")
        score_label.setStyleSheet(f"color: {'green' if score > 0.5 else 'orange'}")

        header_layout.addWidget(filename_label)
        header_layout.addStretch()
        header_layout.addWidget(score_label)

        layout.addLayout(header_layout)

        # Tekstin esikatselu
        text_edit = QTextEdit()
        text_edit.setPlainText(self.chunk_info["text_preview"])
        text_edit.setReadOnly(True)
        text_edit.setMaximumHeight(80)
        text_edit.setStyleSheet("background-color: #f5f5f5; color: #000; border: 1px solid #ddd;")

        layout.addWidget(text_edit)

        self.setLayout(layout)
        self.setStyleSheet("background-color: white; border: 1px solid #ccc; border-radius: 3px;")

class ContextPanel(QWidget):
    """Paneeli kontekstiosien näyttämiseen."""

    def __init__(self, parent=None):
        """
        Parametrit ja attribuutit:
            parent (Any): Vanhempi (parent) widget.
        """
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Asettaa käyttöliittymän komponentit."""
        layout = QVBoxLayout()
        
        # Otsikko
        title_label = QLabel("Haettu konteksti")
        title_label.setStyleSheet("color: #333; font-size: 11pt; font-weight: bold;")
        layout.addWidget(title_label)

        # Vieritettävä alue kontekstiosille
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.chunks_container = QWidget()
        self.chunks_layout = QVBoxLayout()
        self.chunks_layout.addStretch()
        self.chunks_container.setLayout(self.chunks_layout)

        scroll.setWidget(self.chunks_container)
        layout.addWidget(scroll)

        self.setLayout(layout)

    def display_chunks(self, chunks):
        """
        Näyttää haetyt kontekstiosat.

        Parametrit:
            chunks (list[dict]): Lista sanakirjoja, jotka sisältävät kontekstiosien tiedot.
        """
        # Tyhjennetään vanhat osat
        self._clear_chunks()

        # Lisätään uudet osat
        for chunk_info in chunks:
            widget = ContextChunkWidget(chunk_info)
            self.chunks_layout.insertWidget(self.chunks_layout.count() - 1, widget)
        
    def _clear_chunks(self):
        """Poistaa kaikki nykyiset kontekstiosat."""
        while self.chunks_layout.count() > 1:  # Jätä viimeinen väli
            item = self.chunks_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

class ChatPanel(QWidget):
    """
    Chat-käyttöliittymäkomponentti kysymysten esittämiseen ja vastausten näyttämiseen.

    Signaalit:
        question_asked: Lähetetään, kun käyttäjä esittää kysymyksen (question_text).
    """

    question_asked = Signal(str)

    def __init__(self, parent=None):
        """
        Parametrit ja attribuutit:
            parent (Any): Vanhempi (parent) widget.
        """
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Asettaa käyttöliittymän komponentit."""
        layout = QVBoxLayout()

        # Luodaan jakaja keskustelu- ja kontekstipaneelien välillä
        splitter = QSplitter(Qt.Horizontal)

        # Vasen puoli (keskustelu)
        chat_widget = self._create_chat_widget()
        splitter.addWidget(chat_widget)

        # Oikea puoli (kontekstipaneeli)
        self.context_panel = ContextPanel()
        splitter.addWidget(self.context_panel)

        # Asetetaan jakajan suhteelliset koot (60% keskustelu, 40% konteksti)
        splitter.setSizes([600, 400])

        layout.addWidget(splitter)
        self.setLayout(layout)

    def _create_chat_widget(self):
        """
        Luo keskustelupaneelin widgetin.

        Palauttaa:
            QWidget: Keskustelupaneelin widget.
        """
        widget = QWidget()
        layout = QVBoxLayout()

        # Otsikko
        title_label = QLabel("Kysymyksiä ja vastauksia")
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title_label)

        # Keskusteluhistoria
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        self.history_text.setPlaceholderText(
            "Kysy kysymyksiä dokumenteistasi...\n\n"
            "Keskusteluhistoria näkyy tässä"
        )
        layout.addWidget(self.history_text)

        # Kysymyskenttä
        input_layout = QHBoxLayout()

        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText("Kirjoita kysymyksesi tähän englanniksi...")
        self.question_input.returnPressed.connect(self._ask_question)

        self.ask_button = QPushButton("Kysy")
        self.ask_button.clicked.connect(self._ask_question)
        self.ask_button.setMinimumWidth(80)

        input_layout.addWidget(self.question_input)
        input_layout.addWidget(self.ask_button)

        layout.addLayout(input_layout)

        # Tila
        self.status_label = QLabel('')
        self.status_label.setStyleSheet("color: gray; font-size: 9pt;")
        layout.addWidget(self.status_label)

        widget.setLayout(layout)
        return widget
    
    def _ask_question(self):
        """Käsittelee käyttäjän esittämän kysymyksen."""
        question = self.question_input.text().strip()

        if not question:
            return
        
        # Tyhjennetään syötekenttä
        self.question_input.clear()

        # Lähetetään signaali kysymyksestä
        self.question_asked.emit(question)

    def add_question_to_history(self, question):
        """
        Lisää käyttäjän esittämän kysymyksen keskusteluhistoriaan.

        Parametrit:
            question (str): Käyttäjän esittämä kysymys.
        """
        self.history_text.append(
            '<div style="margin: 10px 0;">'
            f'<b style="color: #2196F3;">K: </b> {question}'
            '</div>'
        )

        # Vieritetään alaspäin
        cursor = self.history_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.history_text.setTextCursor(cursor)

    def add_answer_to_history(self, answer, sources=None):
        """
        Lisää mallin antaman vastauksen keskusteluhistoriaan.

        Parametrit:
            answer (str): Vastausteksti.
            sources (list[dict], Optional): Lista sanakirjoja, jotka sisältävät lähdetiedot.
        """
        # Kootaan HTML lähdetietoja varten
        html = '<div style="padding: 10px; margin: 10px 0; background-color: #f5f5f5; border-radius: 5px;">'
        html += f'<b style="color: #4CAF50;">V: </b> {answer}'

        if sources:
            html += '<br><br><i style="color: #666; font-size: 9pt;">Lähteet:</i><br>'
            for i, source in enumerate(sources, 1):
                filename = source.get("doc_filename", "Tuntematon")
                score = source.get("similarity_score", 0)
                html += (
                    '<span style="color: #666; font-size: 9pt;">'
                    f'[{i}] {filename} (pisteet: {score:.3f})</span><br>'
                )

        html += "</div>"

        self.history_text.append(html)

        # Vieritetään alaspäin
        cursor = self.history_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.history_text.setTextCursor(cursor)

    def display_context(self, chunks):
        """
        Näyttää haetut kontekstiosat kontekstipaneelissa.

        Parametrit:
            chunks (list[dict]): Lista sanakirjoja, jotka sisältävät kontekstiosien tiedot.
        """
        self.context_panel.display_chunks(chunks)

    def set_status(self, status):
        """
        Asettaa tilaviestin.

        Parametrit:
            status (str): Tilaviesti.
        """
        self.status_label.setText(status)

    def clear_history(self):
        """Tyhjentää keskusteluhistorian."""
        self.history_text.clear()
        self.context_panel._clear_chunks()
        self.status_label.setText('')

    def set_enabled(self, enabled):
        """
        Asettaa kysymyskentän ja napin käyttöön tai pois käytöstä.

        Parametrit:
            enabled (bool): True, jos komponentit otetaan käyttöön, muuten False.
        """
        self.question_input.setEnabled(enabled)
        self.ask_button.setEnabled(enabled)

    def add_system_message(self, message):
        """
        Lisää järjestelmäviestin keskusteluhistoriaan.

        Parametrit:
            message (str): Järjestelmäviesti.
        """
        self.history_text.append(
            '<div style="margin: 10px 0; color: #FF9800; font-style: italic;">'
            f'Järjestelmä: {message}'
            '</div>'
        )