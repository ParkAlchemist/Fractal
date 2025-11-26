import sys
from PyQt5.QtWidgets import QApplication

from view import MandelbrotViewer

app = QApplication(sys.argv)
viewer = MandelbrotViewer()
viewer.show()
viewer.render_fractal()
sys.exit(app.exec_())
