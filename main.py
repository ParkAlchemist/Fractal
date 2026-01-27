import sys
from PySide6.QtWidgets import QApplication

from ui.view import FractalViewer

app = QApplication(sys.argv)
viewer = FractalViewer()
viewer.show()
viewer.start_render()
sys.exit(app.exec())
