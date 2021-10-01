import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


class UserInterface(QMainWindow):
    def __init__(self):
        super(UserInterface, self).__init__()

        # window title and size
        self.setWindowTitle("Robot Commander")
        self.setGeometry(250, 250, 500, 500)

        # prompt
        self.query = QLabel(self)
        self.query.setText("Select the next robot action:")
        self.query.setFont(QFont('Arial', 14))
        self.query.adjustSize()
        self.query.move(100, 50)

        # inputs
        self.options = []
        self.user_choice = None

    def set_options(self, options):
        option_x = 125
        option_y = 100
        for opt in options:
            self.options.append(QPushButton(self))
            self.options[-1].setText(opt)
            self.options[-1].setFont(QFont('Arial', 12))
            self.options[-1].setGeometry(option_x, option_y, 225, 50)
            self.options[-1].setCheckable(True)
            self.options[-1].clicked.connect(self.set_choice)
            option_y += 50

    def set_choice(self):
        for option in self.options:
            if option.isChecked():
                self.user_choice = option.text()



# def main():
#     app = QApplication(sys.argv)
#     win = UserInterface()
#     win.set_options(["fetch propeller", "fetch wing", "wait"])
#     win.show()
#     sys.exit(app.exec_())
#
#
# main()
