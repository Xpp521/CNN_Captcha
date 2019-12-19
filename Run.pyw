from PyQt4 import QtGui
from UI_Thread.MainWindow import MainWindow


if __name__ == '__main__':
    app = QtGui.QApplication([])
    main_window = MainWindow()
    main_window.show()
    app.exec_()
