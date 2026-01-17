"""
Widget RAG-järjestelmän dokumenttien hallintaan, mukaan lukien
niiden lataamiseen, näyttämiseen ja poistamiseen.
"""

from PySide6.QtWidgets import (
    QListWidgetItem, QWidget, QVBoxLayout, QLabel, QPushButton, 
    QListWidget, QHBoxLayout, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt, Signal
from core.document_loader import Document
from utils.helpers import format_document_size

class DocumentListItem(QListWidgetItem):
    """Listaelementti dokumentille."""

    def __init__(self, document: Document):
        """
        Parametrit ja attribuutit:
            document (Document): Document-olio näytettävälle dokumentille.
        """
        super().__init__()
        self.document = document

        # Muotoillaan näytettävä teksti
        display_text = (
            f"{document.filename}\n"
            f"Koko: {format_document_size(document.file_size)} | Osia: {len(document.chunks)}"
        )

        self.setText(display_text)
        self.setData(Qt.UserRole, document.doc_id)
 
class DocumentPanel(QWidget):
    """
    Widget dokumenttien hallintaan RAG-järjestelmässä.
    
    Signaalit:
        document_uploaded: Lähetetään, kun dokumentti on lisätty (filepath)
        document_deleted: Lähetetään, kun dokumentti on poistettu (doc_id)
        document_selected: Lähetetään, kun dokumentti on valittu (doc_id)
    """

    document_uploaded = Signal(str) # filepath
    document_deleted = Signal(str) # doc_id
    document_selected = Signal(str) # doc_id

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
        title_label = QLabel("Dokumentit")
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title_label)

        # Lataa-nappi
        self.upload_button = QPushButton("Lataa dokumentti")
        self.upload_button.clicked.connect(self._upload_document)
        layout.addWidget(self.upload_button)

        # Dokumenttilista
        self.document_list = QListWidget()
        self.document_list.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.document_list)

        # Muut napit
        button_layout = QHBoxLayout()

        self.delete_button = QPushButton("Poista")
        self.delete_button.clicked.connect(self._delete_document)
        button_layout.addWidget(self.delete_button)

        self.view_button = QPushButton("Näytä tiedot")
        self.view_button.clicked.connect(self._view_document_info)
        button_layout.addWidget(self.view_button)

        button_layout.addWidget(self.delete_button)
        button_layout.addWidget(self.view_button)

        layout.addLayout(button_layout)

        # Statistiikka
        self.stats_label = QLabel("Dokumentteja: 0 | Osat: 0")
        self.stats_label.setStyleSheet("color: gray; font-size: 9pt;")
        layout.addWidget(self.stats_label)

        self.setLayout(layout)

    def _upload_document(self):
        """Käsittelee dokumentin lataamisen."""
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Dokumentit (*.txt *.pdf)")
        file_dialog.setFileMode(QFileDialog.ExistingFile)

        if file_dialog.exec():
            filepaths = file_dialog.selectedFiles()
            for filepath in filepaths:
                # Läheteään signaali dokumentin lataamisesta
                self.document_uploaded.emit(filepath)

    def _delete_document(self):
        """Käsittelee dokumentin poistamisen."""
        current_item = self.document_list.currentItem()
        if not current_item:
            return
        
        # Vahvistetaan poisto
        doc_id = current_item.data(Qt.UserRole)
        document = current_item.document

        reply = QMessageBox.question(
            self,
            "Vahvista poisto",
            f"Haluatko varmasti poistaa dokumentin {document.filename}?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Lähetetään signaali dokumentin poistamisesta
            self.document_deleted.emit(doc_id)

    def _view_document_info(self):
        """Näyttää valitun dokumentin tiedot."""
        curerent_item = self.document_list.currentItem()
        if not curerent_item:
            return
        
        document = curerent_item.document

        info_text = (
            f"Tiedostonimi: {document.filename}\n"
            f"Dokumentin ID: {document.doc_id}\n"
            f"Tiedostokoko: {format_document_size(document.file_size)}\n"
            f"Ladattu: {document.upload_date}\n"
            f"Osien määrä: {len(document.chunks)}\n"
            f"Sisällön merkkimäärä: {len(document.content)} merkkiä"
        )

        QMessageBox.information(self, "Dokumentin tiedot", info_text)

    def _on_selection_changed(self):
        """Käsittelee dokumentin valinnan muutoksen."""
        has_selection = self.document_list.currentItem() is not None
        self.delete_button.setEnabled(has_selection)
        self.view_button.setEnabled(has_selection)

        if has_selection:
            doc_id = self.document_list.currentItem().data(Qt.UserRole)
            self.document_selected.emit(doc_id)

    def add_document(self, document):
        """
        Lisää dokumentin listaan.

        Parametrit:
            document (Document): Lisättävä Document-olio.
        """
        item = DocumentListItem(document)
        self.document_list.addItem(item)
        self._update_statistics()

    def remove_document(self, doc_id):
        """
        Poistaa dokumentin listasta sen ID:n perusteella.

        Parametrit:
            doc_id (str): Poistettavan dokumentin ID.
        """
        for i in range(self.document_list.count()):
            item = self.document_list.item(i)
            if item.data(Qt.UserRole) == doc_id:
                self.document_list.takeItem(i)
                break

        self._update_statistics()

    def get_document_count(self):
        """
        Palauttaa dokumenttien määrän listassa.

        Palauttaa:
            int: Dokumenttien määrä.
        """
        return self.document_list.count()
    
    def _update_statistics(self):
        """Päivittää dokumenttien tilastotiedot."""
        doc_count = self.get_document_count()
        chunk_count = sum(
            len(self.document_list.item(i).document.chunks)
            for i in range(doc_count)
        )

        self.stats_label.setText(
            f"Dokumentteja: {doc_count} | Osat: {chunk_count}"
        )