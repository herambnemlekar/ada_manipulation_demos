import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


class UserInterface(QMainWindow):
    def __init__(self):
        super(UserInterface, self).__init__()

        # window title and size
        self.setWindowTitle("Robot Commander")
        self.setGeometry(0, 0, 1280, 720)

        # prompt
        self.query = QLabel(self)
        self.query.setText("Select the next robot actions:")
        self.query.setFont(QFont('Arial', 14))
        self.query.adjustSize()
        self.query.move(120, 50)

        # task info
        self.image = QLabel(self)
        pixmap = QPixmap('task.jpg')
        self.image.setPixmap(pixmap)
        self.image.adjustSize()
        self.image.move(450, 50)

        # inputs
        self.options = []
        self.user_choice = []
        self.act = False

    def set_options(self, options, suggestions=[]):
        option_x = 125
        option_y = 100
        buttons = []
        for opt in options:
            buttons.append(QPushButton(self))
            buttons[-1].setText(opt)
            buttons[-1].setFont(QFont('Arial', 12))
            buttons[-1].setGeometry(option_x, option_y, 225, 50)
            buttons[-1].setCheckable(True)
            # buttons[-1].clicked.connect(self.set_choice)
            if opt in suggestions:
                buttons[-1].setStyleSheet("QPushButton {background-color : lightgreen;} QPushButton::checked {background-color : lightpink;}")
            else:
                buttons[-1].setStyleSheet("QPushButton::checked {background-color : lightpink;}")
            option_y += 50    
        self.options = buttons

        option_x = 100
        option_y += 50
        self.suggested_button = QPushButton(self)
        self.suggested_button.setText("Perform the SUGGESTED actions")
        self.suggested_button.setFont(QFont('Arial', 12))
        self.suggested_button.setGeometry(option_x, option_y, 275, 50)
        self.suggested_button.setStyleSheet("background-color : lightgreen")
        self.suggested_button.setCheckable(True)
        self.suggested_button.clicked.connect(self.set_choice)

        option_x = 100
        option_y += 75
        self.selected_button = QPushButton(self)
        self.selected_button.setText("Perform the SELECTED actions")
        self.selected_button.setFont(QFont('Arial', 12))
        self.selected_button.setGeometry(option_x, option_y, 275, 50)
        self.selected_button.setStyleSheet("background-color : lightpink")
        self.selected_button.setCheckable(True)
        self.selected_button.clicked.connect(self.set_choice)

    def set_choice(self):
        self.act = True
        
        if self.selected_button.isChecked():
            for option in self.options:
                if option.isChecked():
                    self.user_choice.append(option.text())
                    # option.toggle()

        self.close()


# def main():
#     app = QApplication(sys.argv)
#     win = UserInterface()
#     win.set_options(["fetch propeller", "fetch wing", "wait"])
#     win.show()
#     sys.exit(app.exec_())
#
#
# main()
