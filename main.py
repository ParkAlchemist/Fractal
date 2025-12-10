import sys

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from view import FractalViewer

app = QApplication(sys.argv)
viewer = FractalViewer()
viewer.show()
QTimer.singleShot(0, viewer.render_fractal)
sys.exit(app.exec_())
