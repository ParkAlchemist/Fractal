import sys
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from ui.view import FractalViewer

app = QApplication(sys.argv)
viewer = FractalViewer()
viewer.show()
QTimer.singleShot(0, viewer.render_fractal)
sys.exit(app.exec())
