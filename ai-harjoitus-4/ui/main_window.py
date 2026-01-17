"""
Pääikkuna, joka integroi kaikki komponentit, mukaan lukien
dokumenttien hallinnan, chat-käyttöliittymän ja asetukset.
"""

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QMainWindow, QMessageBox, QWidget, QHBoxLayout, QSplitter,
    QStatusBar, QProgressBar
)
from PySide6.QtGui import QAction
import os

from core.document_loader import Document, DocumentLoader
from core.embedder import Embedder, EmbeddingCache
from core.retriever import Retriever
from core.llm_client import LLMClient
from utils.helpers import Config, ensure_directory_exists
from ui.document_panel import DocumentPanel
from ui.chat_panel import ChatPanel
from ui.settings_dialog import SettingsDialog, ModelInfoDialog

class DocumentProcessingThread(QThread):
    """Taustasäie dokumenttien käsittelyyn."""

    progress = Signal(str) # Tilaviesti
    finished = Signal(Document, object) # Dokumentti ja upotukset
    error = Signal(str) # Virheviesti

    def __init__(self, filepath, loader: DocumentLoader, embedder: Embedder):
        """
        Parametrit ja attribuutit:
            filepath (str): Tiedoston polku.
            loader (DocumentLoader): Dokumentin lataaja.
            embedder (Embedder): Upotusten luoja.       
        """
        super().__init__()
        self.filepath = filepath
        self.loader = loader
        self.embedder = embedder

    def run(self):
        """Aloittaa dokumentin käsittelyn."""
        try:
            # Ladataan dokumentti
            self.progress.emit(f"Ladataan dokumentti: {os.path.basename(self.filepath)}")
            document = self.loader.create_document(self.filepath)

            # Luodaan upotukset
            self.progress.emit(f"Luodaan {len(document.chunks)} osaa upotuksille...")
            chunk_texts = [chunk.text for chunk in document.chunks]
            embeddings = self.embedder.embed_batch(chunk_texts, batch_size=32)

            self.progress.emit("Dokumentti käsitelty onnistuneesti!")
            self.finished.emit(document, embeddings)

        except Exception as e:
            self.error.emit(f"Virhe dokumentin käsittelyssä: {str(e)}")

class QuestionAnsweringThread(QThread):
    """Taustasäie kysymysten käsittelyyn."""

    progress = Signal(str) # Tilaviesti
    finished = Signal(dict) # Tulossanakirja
    error = Signal(str) # Virheviesti

    def __init__(self, query, retriever: Retriever, llm_client: LLMClient, top_k, max_context):
        """
        Parametrit ja attribuutit:
            query (str): Käyttäjän kysymys.
            retriever (Retriever): Retriever-instanssi.
            llm_client (LLMClient): LLMClient-instanssi.
            top_k (int): Haettavien osien määrä.
            max_context (int): Maksimi kontekstin pituus.       
        """
        super().__init__()
        self.query = query
        self.retriever = retriever
        self.llm_client = llm_client
        self.top_k = top_k
        self.max_context = max_context

    def run(self):
        """Aloittaa kysymyksen käsittelyn."""
        try:
            # Haetaan relevantit osat
            self.progress.emit("Haetaan relevanttia tietoa...")
            results = self.retriever.retrieve(self.query, top_k=self.top_k)

            # Luodaan vastaus
            self.progress.emit("Mietitään vastausta...")
            answer_data = self.llm_client.generate_answer(
                self.query, results, max_context_length=self.max_context
            )

            self.progress.emit("Vastaus saatu!")
            self.finished.emit(answer_data)

        except Exception as e:
            self.error.emit(f"Virhe kysymyksen käsittelyssä: {str(e)}")

class MainWindow(QMainWindow):
    """Pääikkuna RAG-järjestelmän käyttöliittymälle."""

    def __init__(self):
        super().__init__()

        # Konfiguraatio
        self.config = Config()
        self.config_file = "data/configuration.json"
        self._load_config()

        # Alustetaan komponentit
        self._init_components()

        # Asetetaan käyttöliittymä
        self._setup_ui()
        self._create_menu_bar()

        # Alustetaan mallit
        self._init_models()

        # Ikkunan määritykset
        self.setWindowTitle("RAG-järjestelmä")
        self.setGeometry(100, 100, 1200, 800)

    def _load_config(self):
        """Lataa konfiguraatio tiedostosta."""
        ensure_directory_exists("data")
        if os.path.exists(self.config_file):
            try:
                self.config.load(self.config_file)
            except Exception as e:
                print(f"Virhe konfiguraation lataamisessa: {str(e)}")

    def _save_config(self):
        """Tallentaa konfiguraation tiedostoon."""
        try:
            self.config.save(self.config_file)
        except Exception as e:
            print(f"Virhe konfiguraation tallentamisessa: {str(e)}")
        

    def _init_components(self):
        """Alustaa ydinkomponentit."""
        self.document_loader = DocumentLoader(
            chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap
        )
        self.embedding_cache = EmbeddingCache()

        # Nämä alustetaan _init_models-funktiossa
        self.embedder = None
        self.retriever = None
        self.llm_client = None

    def _init_models(self):
        """Alustaa koneoppimismallit (upotusmalli ja LLM)."""
        # Näytetään latausviesti
        self.statusBar().showMessage("Ladataan malleja...")

        try:
            # Alustetaan upotusmalli
            self.embedder = Embedder(model_name=self.config.model_name)

            # Alustetaan hakija
            self.retriever = Retriever(
                embedder=self.embedder, top_k=self.config.top_k_chunks
            )

            # Alustetaan LLM
            self.llm_client = LLMClient(
                model_name=self.config.llm_model_name, max_length=512
            )

            self.statusBar().showMessage("Mallit ladattu onnistuneesti", 3000)
            self.chat_panel.set_enabled(True)

        except Exception as e:
            error_message = f"Virhe mallien alustamisessa: {str(e)}"
            QMessageBox.critical(self, "Mallien latausvirhe", error_message)
            self.statusBar().showMessage(error_message, 2000)
            self.chat_panel.set_enabled(False)

    def _setup_ui(self):
        """Asettaa käyttöliittymän komponentit."""
        # Keskimmäinen widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Päänäkymä
        main_layout = QHBoxLayout()

        # Luodaan jakaja
        splitter = QSplitter(Qt.Horizontal)

        # Vasen puoli (dokumenttien hallinta)
        self.document_panel = DocumentPanel()
        self.document_panel.document_uploaded.connect(self._handle_document_upload)
        self.document_panel.document_deleted.connect(self._handle_document_deletion)
        splitter.addWidget(self.document_panel)

        # Oikea puoli (chat-käyttöliittymä)
        self.chat_panel = ChatPanel()
        self.chat_panel.question_asked.connect(self._handle_question_submission)
        self.chat_panel.set_enabled(False) # Poistetaan käytöstä, kunnes mallit on ladattu
        splitter.addWidget(self.chat_panel)

        # Asetetaan jakajan suhteelliset koot (30% dokumentit, 70% chat)
        splitter.setSizes([300, 700])

        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)

        # Tilapalkki
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Edistymispalkki
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.hide()
        self.status_bar.addPermanentWidget(self.progress_bar)
        
    def _create_menu_bar(self):
        """Luo valikkopalkin."""
        menubar = self.menuBar()

        # Tiedostovalikko
        file_menu = menubar.addMenu("Tiedosto")

        upload_action = QAction("Lataa dokumentti", self)
        upload_action.triggered.connect(self.document_panel.upload_button.click)
        file_menu.addAction(upload_action)

        file_menu.addSeparator()

        exit_action = QAction("Poistu", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Muokkausvalikko
        edit_menu = menubar.addMenu("Muokkaa")

        clear_history_action = QAction("Tyhjennä keskusteluhistoria", self)
        clear_history_action.triggered.connect(self.chat_panel.clear_history)
        edit_menu.addAction(clear_history_action)

        settings_action = QAction("Asetukset", self)
        settings_action.triggered.connect(self._show_settings_dialog)
        edit_menu.addAction(settings_action)

        # Näytä-valikko
        view_menu = menubar.addMenu("Näytä")

        stats_action = QAction("Näytä tilastot", self)
        stats_action.triggered.connect(self._show_statistics)
        view_menu.addAction(stats_action)

        model_info_action = QAction("Näytä mallien tiedot", self)
        model_info_action.triggered.connect(self._show_model_info)
        view_menu.addAction(model_info_action)

        # Ohje-valikko
        help_menu = menubar.addMenu("Ohje")

        about_action = QAction("Tietoja", self)
        about_action.triggered.connect(self._show_about_dialog)
        help_menu.addAction(about_action)

    def _handle_document_upload(self, filepath):
        """
        Käsittelee dokumentin lataamisen.
        
        Parametrit:
            filepath (str): Ladatun tiedoston polku.
        """
        # Otetaan lataaminen pois käytöstä, kun dokumenttia käsitellään
        self.document_panel.upload_button.setEnabled(False)
        self.progress_bar.show()
        self.progress_bar.setRange(0, 0)

        # Luodaan käsittelysäie
        self.processing_thread = DocumentProcessingThread(
            filepath, self.document_loader, self.embedder
        )
        self.processing_thread.progress.connect(self._update_status)
        self.processing_thread.finished.connect(self._document_processing_finished)
        self.processing_thread.error.connect(self._processing_error)
        self.processing_thread.start()
        
    def _document_processing_finished(self, document: Document, embeddings):
        """
        Käsittelee onnistuneen dokumentin käsittelyn.
        
        Parametrit:
            document (Document): Käsitelty dokumentti.
            embeddings (np.ndarray): Dokumentin osien upotukset.
        """
        # Lisätään dokumentti hakijalle
        self.retriever.add_document(document, embeddings)

        # Lisätään upotukset välimuistiin
        self.embedding_cache.save_embeddings(document.doc_id, embeddings)

        # Lisätään käyttöliittymään
        self.document_panel.add_document(document)

        # Otetaan lataaminen takaisin käyttöön
        self.document_panel.upload_button.setEnabled(True)
        self.progress_bar.hide()

        self.statusBar().showMessage(f"Dokumentti {document.filename} ladattu onnistuneesti", 3000)

    def _processing_error(self, error_message):
        """
        Käsittelee virheen dokumentin käsittelyssä.

        Parametrit:
            error_message (str): Virheviesti.
        """
        QMessageBox.warning(self, "Käsittelyvirhe", error_message)
        self.document_panel.upload_button.setEnabled(True)
        self.progress_bar.hide()
        self.statusBar().showMessage(error_message, 2000)        

    def _handle_document_deletion(self, doc_id):
        """
        Käsittelee dokumentin poistamisen.

        Parametrit:
            doc_id (str): Poistettavan dokumentin tunniste.
        """
        # Poistetaan dokumentti hakijalta
        self.retriever.remove_document(doc_id)

        # Poistetaan upotukset välimuistista
        self.embedding_cache.delete_embeddings(doc_id)

        # Poistetaan käyttöliittymästä
        self.document_panel.remove_document(doc_id)

        self.statusBar().showMessage("Dokumentti poistettu onnistuneesti", 2000)

    def _handle_question_submission(self, question):
        """
        Käsittelee käyttäjän esittämän kysymyksen.

        Parametrit:
            question (str): Käyttäjän kysymys.
        """
        # Tarkistetaan, onko dokumentteja ladattu
        if self.retriever.get_document_count() == 0:
            self.chat_panel.add_system_message("Lataa dokumentteja ennen kysymysten esittämistä.")
            return
        
        # Lisätään kysymys historiaan
        self.chat_panel.add_question_to_history(question)

        # Otetaan chat käyttöliittymästä pois käytöstä
        self.chat_panel.set_enabled(False)
        self.progress_bar.show()
        self.progress_bar.setRange(0, 0)

        # Luodaan kysymyksen käsittelysäie
        self.qa_thread = QuestionAnsweringThread(
            question, self.retriever, self.llm_client, 
            self.config.top_k_chunks, self.config.max_context_length
        )
        self.qa_thread.progress.connect(self._update_status)
        self.qa_thread.finished.connect(self._answer_generated)
        self.qa_thread.error.connect(self._qa_error)
        self.qa_thread.start()

    def _answer_generated(self, answer_data):
        """
        Käsittelee onnistuneen vastauksen luonnin.

        Parametrit:
            answer_data (dict): Vastaussanakirja.
        """
        # Lisätään vastaus historiaan
        self.chat_panel.add_answer_to_history(answer_data["answer"], answer_data["sources"])

        # Näytetään konteksti
        self.chat_panel.display_context(answer_data["sources"])

        # Otetaan chat käyttöliittymään takaisin käyttöön
        self.chat_panel.set_enabled(True)
        self.progress_bar.hide()
        self.statusBar().showMessage("Vastaus saatu!", 2000)

    def _qa_error(self, error_message):
        """
        Käsittelee virheen kysymyksen käsittelyssä.

        Parametrit:
            error_message (str): Virheviesti.
        """
        self.chat_panel.add_system_message(f"{error_message}")
        self.chat_panel.set_enabled(True)
        self.progress_bar.hide()
        self.statusBar().showMessage(error_message, 2000)
        
    def _update_status(self, message):
        """
        Päivittää tilapalkin viestin.

        Parametrit:
            message (str): Näytettävä viesti.
        """
        self.statusBar().showMessage(message, 3000)

    def _show_settings_dialog(self):
        """Näyttää asetukset-dialogin."""
        dialog = SettingsDialog(self.config, self)
        dialog.settings_changed.connect(self._apply_settings)
        dialog.exec()

    def _apply_settings(self, new_config: Config):
        """
        Ottaa uudet asetukset käyttöön.

        Parametrit:
            new_config (Config): Uusi konfiguraatio.
        """
        self.config = new_config

        # Päivitetään komponentit
        self.document_loader.update_chunk_size(self.config.chunk_size, self.config.chunk_overlap)
        self.retriever.update_top_k(self.config.top_k_chunks)

        # Tallennetaan konfiguraatio
        self._save_config()

        self.statusBar().showMessage("Asetukset päivitetty", 2000)
        

    def _show_statistics(self):
        """Näyttää tilastotietoja."""
        stats = self.retriever.get_statistics()

        stats_text = (
            f"Dokumentteja: {stats['document_count']}\n"
            f"Osia: {stats['chunk_count']}\n"
            f"Osia keskimäärin per dokumentti: {stats['average_chunks_per_document']:.1f}\n"
            f"Upotusulottuvuus: {stats['embedding_dimension']}\n"
        )

        QMessageBox.information(self, "Tilastotiedot", stats_text)

    def _show_model_info(self):
        """Näyttää mallien tiedot."""
        if not self.embedder or not self.llm_client:
            QMessageBox.warning(self, "Malleja ei ladattu", "Malleja ei oltu vielä ladattu")
            return
        
        embedder_info = {
            "model_name": self.embedder.model_name,
            "device": str(self.embedder.device),
            "embedding_dimension": self.embedder.get_embedding_dimension()
        }

        llm_info = self.llm_client.get_model_info()

        dialog = ModelInfoDialog(embedder_info, llm_info, self)
        dialog.exec()

    def _show_about_dialog(self):
        """Näyttää tietoja sovelluksesta."""
        about_text = (
            "<h2>RAG-järjestelmä</h2>"
            "<p>Hakutoimintoon perustuva kysymys-vastausjärjestelmä</p>"
            "<p><b>Toiminnot:</b></p>"
            "<ul>"
            "<li>Dokumenttien lataus ja käsittely (txt/pdf)</li>"
            "<li>Semanttinen haku upotuksia hyödyntäen</li>"
            "<li>Määriteltävissä oleva pilkkominen ja hakeminen</li>"
            "</ul>"
            "<p><b>Teknologiat:</b></p>"
            "<ul>"
            "<li>PySide6 käyttöliittymään</li>"
            "<li>HuggingFace-mallit (PyTorch, transformers)</li>"
            "</ul>"
        )

        QMessageBox.about(self, "Tietoja RAG-järjestelmästä", about_text)

    def closeEvent(self, event):
        """Käsittelee ikkunan sulkemisen tapahtuman."""
        self._save_config()
        event.accept()