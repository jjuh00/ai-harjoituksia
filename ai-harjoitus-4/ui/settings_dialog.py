"""
Valintaikkuna RAG-järjestelmän parametreille, kuten
osan kokoluokalle, hettujen osien määrälle ja mallin valinnalle.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QGroupBox, 
    QFormLayout, QSpinBox, QLabel
)
from PySide6.QtCore import Signal
from utils.helpers import Config

class SettingsDialog(QDialog):
    """
    Dialogi RAG-järjestelmän asetusten muokkaamiseen.
    Lähettää settings_changed-signaalin, kun asetukset on tallennettu.
    """
    
    settings_changed = Signal(Config)

    def __init__(self, config: Config, parent=None):
        """
        Parametrit ja attribuutit:
            config (Config): Config-olio, sisältää konfiguraatioasetukset.
            parent (Any): Vanhempi (parent) widget.
        """
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Asetukset")
        self.setModal(True)
        self.setMinimumWidth(400)

        self._setup_ui()
        self._load_current_settings()

    def _setup_ui(self):
        """
        Asettaa käyttöliittymän komponentit.
        """
        layout = QVBoxLayout()

        # Dokumentin käsittelyn asetukset
        doc_group = self._create_document_settings_group()
        layout.addWidget(doc_group)

        # Hakuasetukset
        retrieval_group = self._create_retrieval_settings_group()
        layout.addWidget(retrieval_group)

        # Luontiasetukset
        generation_group = self._create_generation_settings_group()
        layout.addWidget(generation_group)

        # Napit
        button_layout = QHBoxLayout()

        self.save_button = QPushButton("Tallenna")
        self.save_button.clicked.connect(self._save_settings)

        self.cancel_button = QPushButton("Peruuta")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _create_document_settings_group(self):
        """
        Luo dokumentin käsittelyn asetusten ryhmän.

        Palauttaa:
            QGroupBox: Dokumentin käsittelyn asetusten ryhmä.
        """
        group = QGroupBox("Dokumentin käsittely")
        form_layout = QFormLayout()

        # Osan koko
        self.chunk_size_spin = QSpinBox()
        self.chunk_size_spin.setMinimum(100)
        self.chunk_size_spin.setMaximum(2000)
        self.chunk_size_spin.setSingleStep(50)
        self.chunk_size_spin.setSuffix(" merkkiä")
        form_layout.addRow("Osan koko:", self.chunk_size_spin)

        # Osien päällekkäisyys
        self.chunk_overlap_spin = QSpinBox()
        self.chunk_overlap_spin.setMinimum(0)
        self.chunk_overlap_spin.setMaximum(500)
        self.chunk_overlap_spin.setSingleStep(10)
        self.chunk_overlap_spin.setSuffix(" merkkiä")
        form_layout.addRow("Osien päällekkäisyys:", self.chunk_overlap_spin)

        group.setLayout(form_layout)
        return group

    def _create_retrieval_settings_group(self):
        """
        Luo hakuasetusten ryhmän.

        Palauttaa:
            QGroupBox: Hakuasetusten ryhmä.
        """
        group = QGroupBox("Hakuasetukset")
        form_layout = QFormLayout()

        # Haettavien osien määrä
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setMinimum(1)
        self.top_k_spin.setMaximum(10)
        self.top_k_spin.setSingleStep(1)
        self.top_k_spin.setSuffix(" osaa")
        form_layout.addRow("Haettavien osien määrä:", self.top_k_spin)

        group.setLayout(form_layout)
        return group
    
    def _create_generation_settings_group(self):
        """
        Luo luontiasetusten ryhmän.

        Palauttaa:
            QGroupBox: Luontiasetusten ryhmä.
        """
        group = QGroupBox("Vastauksen luonti")
        form_layout = QFormLayout()

        # Maksimi kontekstin pituus
        self.max_context_spin = QSpinBox()
        self.max_context_spin.setMinimum(512)
        self.max_context_spin.setMaximum(4096)
        self.max_context_spin.setSingleStep(256)
        self.max_context_spin.setSuffix(" merkkiä")
        form_layout.addRow("Maksimi kontekstin pituus:", self.max_context_spin)

        group.setLayout(form_layout)
        return group
    
    def _load_current_settings(self):
        """Lataa nykyiset asetukset Config-oliosta käyttöliittymään."""
        self.chunk_size_spin.setValue(self.config.chunk_size)
        self.chunk_overlap_spin.setValue(self.config.chunk_overlap)
        self.top_k_spin.setValue(self.config.top_k_chunks)
        self.max_context_spin.setValue(self.config.max_context_length)

    def _save_settings(self):
        """Tallentaa asetukset Config-olioon ja lähettää signaalin."""
        # Luodaan uusi Connfig-olio päivitetyillä arvoilla
        new_config = Config()
        new_config.chunk_size = self.chunk_size_spin.value()
        new_config.chunk_overlap = self.chunk_overlap_spin.value()
        new_config.top_k_chunks = self.top_k_spin.value()
        new_config.max_context_length = self.max_context_spin.value()

        # Kopioidana muut asetukset vanhasta konfiguraatiosta
        new_config.model_name = self.config.model_name
        new_config.llm_model_name = self.config.llm_model_name
        new_config.data_directory = self.config.data_directory

        # Lähetetään signaali uusilla asetuksilla
        self.settings_changed.emit(new_config)
        self.accept()
    
class ModelInfoDialog(QDialog):
    """
    Dialogi mallin tiedoille.
    Näyttää valitun upotusmallin ja LLM-mallin tiedot.
    """

    def __init__(self, embedder_info, llm_info, parent=None):
        """
        Parametrit ja attribuutit:
            embedder_info (dict): Upotusmallin tiedot.
            llm_info (dict): LLM-mallin tiedot.
            parent (Any): Vanhempi (parent) widget.
        """
        super().__init__(parent)
        self.embedder_info = embedder_info
        self.llm_info = llm_info

        self.setWindowTitle("Mallin tiedot")
        self.setModal(True)
        self.setMinimumWidth(400)

        self._setup_ui()

    def _setup_ui(self):
        """Asettaa käyttöliittymän komponentit."""
        layout = QVBoxLayout()

        # Upotusmallin tiedot
        embedder_group = QGroupBox("Upotusmalli")
        embedder_layout = QFormLayout()

        embedder_layout.addRow("Mallin nimi:", QLabel(self.embedder_info.get("model_name", "n/a")))
        embedder_layout.addRow("Laite: ", QLabel(self.embedder_info.get("device", "n/a")))
        embedder_layout.addRow("Upotusulottuvuus:",
                             QLabel(str(self.embedder_info.get("embedding_dimension", "n/a"))))

        embedder_group.setLayout(embedder_layout)
        layout.addWidget(embedder_group)

        # LLM-mallin tiedot
        llm_group = QGroupBox("Kielimalli (LLM)")
        llm_layout = QFormLayout()

        llm_layout.addRow("Mallin nimi:", QLabel(self.llm_info.get("model_name", "n/a")))
        llm_layout.addRow("Laite: ", QLabel(self.llm_info.get("device", "n/a")))
        llm_layout.addRow("Maksimi syötepituus:",
                        QLabel(str(self.llm_info.get("max_length", "n/a"))))

        llm_group.setLayout(llm_layout)
        layout.addWidget(llm_group)

        # Sulje-nappi
        close_button = QPushButton("Sulje")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

        self.setLayout(layout)