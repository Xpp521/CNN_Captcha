# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow_UI.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_CNN_Captcha(object):
    def setupUi(self, CNN_Captcha):
        CNN_Captcha.setObjectName(_fromUtf8("CNN_Captcha"))
        CNN_Captcha.resize(580, 576)
        CNN_Captcha.setMaximumSize(QtCore.QSize(580, 576))
        self.centralwidget = QtGui.QWidget(CNN_Captcha)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.tabWidget = QtGui.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 10, 561, 531))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.tabWidget.setFont(font)
        self.tabWidget.setInputMethodHints(QtCore.Qt.ImhNone)
        self.tabWidget.setObjectName(_fromUtf8("tabWidget"))
        self.tab = QtGui.QWidget()
        self.tab.setObjectName(_fromUtf8("tab"))
        self.groupBox = QtGui.QGroupBox(self.tab)
        self.groupBox.setGeometry(QtCore.QRect(10, 310, 531, 171))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.label = QtGui.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(30, 40, 100, 30))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName(_fromUtf8("label"))
        self.lineEdit = QtGui.QLineEdit(self.groupBox)
        self.lineEdit.setGeometry(QtCore.QRect(110, 40, 291, 30))
        self.lineEdit.setReadOnly(True)
        self.lineEdit.setObjectName(_fromUtf8("lineEdit"))
        self.pushButton = QtGui.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(420, 40, 100, 30))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.label_2 = QtGui.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(30, 100, 100, 30))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setFrameShape(QtGui.QFrame.NoFrame)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.lineEdit_2 = QtGui.QLineEdit(self.groupBox)
        self.lineEdit_2.setGeometry(QtCore.QRect(110, 100, 291, 30))
        self.lineEdit_2.setInputMethodHints(QtCore.Qt.ImhUrlCharactersOnly)
        self.lineEdit_2.setObjectName(_fromUtf8("lineEdit_2"))
        self.groupBox_2 = QtGui.QGroupBox(self.tab)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 10, 531, 301))
        self.groupBox_2.setTitle(_fromUtf8(""))
        self.groupBox_2.setObjectName(_fromUtf8("groupBox_2"))
        self.label_3 = QtGui.QLabel(self.groupBox_2)
        self.label_3.setGeometry(QtCore.QRect(140, 220, 100, 30))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.label_4 = QtGui.QLabel(self.groupBox_2)
        self.label_4.setGeometry(QtCore.QRect(230, 220, 100, 30))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.pushButton_3 = QtGui.QPushButton(self.groupBox_2)
        self.pushButton_3.setEnabled(False)
        self.pushButton_3.setGeometry(QtCore.QRect(290, 220, 110, 30))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName(_fromUtf8("pushButton_3"))
        self.pushButton_2 = QtGui.QPushButton(self.groupBox_2)
        self.pushButton_2.setEnabled(False)
        self.pushButton_2.setGeometry(QtCore.QRect(210, 260, 100, 30))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        self.label_7 = QtGui.QLabel(self.groupBox_2)
        self.label_7.setGeometry(QtCore.QRect(0, 0, 531, 211))
        self.label_7.setFrameShape(QtGui.QFrame.Box)
        self.label_7.setText(_fromUtf8(""))
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.tabWidget.addTab(self.tab, _fromUtf8(""))
        self.tab_2 = QtGui.QWidget()
        self.tab_2.setObjectName(_fromUtf8("tab_2"))
        self.groupBox_3 = QtGui.QGroupBox(self.tab_2)
        self.groupBox_3.setGeometry(QtCore.QRect(10, 10, 531, 121))
        self.groupBox_3.setObjectName(_fromUtf8("groupBox_3"))
        self.pushButton_6 = QtGui.QPushButton(self.groupBox_3)
        self.pushButton_6.setEnabled(False)
        self.pushButton_6.setGeometry(QtCore.QRect(370, 80, 100, 30))
        self.pushButton_6.setObjectName(_fromUtf8("pushButton_6"))
        self.pushButton_4 = QtGui.QPushButton(self.groupBox_3)
        self.pushButton_4.setGeometry(QtCore.QRect(50, 80, 100, 30))
        self.pushButton_4.setObjectName(_fromUtf8("pushButton_4"))
        self.pushButton_5 = QtGui.QPushButton(self.groupBox_3)
        self.pushButton_5.setEnabled(False)
        self.pushButton_5.setGeometry(QtCore.QRect(210, 80, 100, 30))
        self.pushButton_5.setObjectName(_fromUtf8("pushButton_5"))
        self.label_5 = QtGui.QLabel(self.groupBox_3)
        self.label_5.setGeometry(QtCore.QRect(30, 30, 81, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_5.setFont(font)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.comboBox = QtGui.QComboBox(self.groupBox_3)
        self.comboBox.setGeometry(QtCore.QRect(120, 30, 88, 30))
        self.comboBox.setObjectName(_fromUtf8("comboBox"))
        self.comboBox.addItem(_fromUtf8(""))
        self.comboBox.addItem(_fromUtf8(""))
        self.comboBox.addItem(_fromUtf8(""))
        self.comboBox.addItem(_fromUtf8(""))
        self.comboBox.addItem(_fromUtf8(""))
        self.comboBox.addItem(_fromUtf8(""))
        self.comboBox.addItem(_fromUtf8(""))
        self.label_6 = QtGui.QLabel(self.groupBox_3)
        self.label_6.setGeometry(QtCore.QRect(240, 30, 150, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_6.setFont(font)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.comboBox_2 = QtGui.QComboBox(self.groupBox_3)
        self.comboBox_2.setGeometry(QtCore.QRect(390, 30, 88, 30))
        self.comboBox_2.setObjectName(_fromUtf8("comboBox_2"))
        self.comboBox_2.addItem(_fromUtf8(""))
        self.comboBox_2.addItem(_fromUtf8(""))
        self.comboBox_2.addItem(_fromUtf8(""))
        self.comboBox_2.addItem(_fromUtf8(""))
        self.comboBox_2.addItem(_fromUtf8(""))
        self.label_8 = QtGui.QLabel(self.groupBox_3)
        self.label_8.setGeometry(QtCore.QRect(480, 30, 20, 30))
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.groupBox_4 = QtGui.QGroupBox(self.tab_2)
        self.groupBox_4.setGeometry(QtCore.QRect(10, 140, 531, 341))
        self.groupBox_4.setObjectName(_fromUtf8("groupBox_4"))
        self.listWidget = QtGui.QListWidget(self.groupBox_4)
        self.listWidget.setGeometry(QtCore.QRect(0, 20, 531, 321))
        self.listWidget.setObjectName(_fromUtf8("listWidget"))
        self.groupBox_4.raise_()
        self.groupBox_3.raise_()
        self.tabWidget.addTab(self.tab_2, _fromUtf8(""))
        self.tab_3 = QtGui.QWidget()
        self.tab_3.setObjectName(_fromUtf8("tab_3"))
        self.groupBox_5 = QtGui.QGroupBox(self.tab_3)
        self.groupBox_5.setGeometry(QtCore.QRect(10, 10, 531, 121))
        self.groupBox_5.setObjectName(_fromUtf8("groupBox_5"))
        self.pushButton_8 = QtGui.QPushButton(self.groupBox_5)
        self.pushButton_8.setEnabled(False)
        self.pushButton_8.setGeometry(QtCore.QRect(280, 80, 100, 30))
        self.pushButton_8.setObjectName(_fromUtf8("pushButton_8"))
        self.pushButton_7 = QtGui.QPushButton(self.groupBox_5)
        self.pushButton_7.setGeometry(QtCore.QRect(140, 80, 100, 30))
        self.pushButton_7.setObjectName(_fromUtf8("pushButton_7"))
        self.label_9 = QtGui.QLabel(self.groupBox_5)
        self.label_9.setGeometry(QtCore.QRect(170, 30, 81, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_9.setFont(font)
        self.label_9.setObjectName(_fromUtf8("label_9"))
        self.comboBox_3 = QtGui.QComboBox(self.groupBox_5)
        self.comboBox_3.setGeometry(QtCore.QRect(260, 30, 88, 30))
        self.comboBox_3.setObjectName(_fromUtf8("comboBox_3"))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.comboBox_3.addItem(_fromUtf8(""))
        self.groupBox_6 = QtGui.QGroupBox(self.tab_3)
        self.groupBox_6.setGeometry(QtCore.QRect(10, 140, 531, 341))
        self.groupBox_6.setObjectName(_fromUtf8("groupBox_6"))
        self.listWidget_2 = QtGui.QListWidget(self.groupBox_6)
        self.listWidget_2.setGeometry(QtCore.QRect(0, 20, 531, 321))
        self.listWidget_2.setObjectName(_fromUtf8("listWidget_2"))
        self.tabWidget.addTab(self.tab_3, _fromUtf8(""))
        self.tab_4 = QtGui.QWidget()
        self.tab_4.setObjectName(_fromUtf8("tab_4"))
        self.groupBox_7 = QtGui.QGroupBox(self.tab_4)
        self.groupBox_7.setGeometry(QtCore.QRect(10, 10, 531, 121))
        self.groupBox_7.setObjectName(_fromUtf8("groupBox_7"))
        self.pushButton_10 = QtGui.QPushButton(self.groupBox_7)
        self.pushButton_10.setEnabled(False)
        self.pushButton_10.setGeometry(QtCore.QRect(280, 80, 100, 30))
        self.pushButton_10.setObjectName(_fromUtf8("pushButton_10"))
        self.pushButton_9 = QtGui.QPushButton(self.groupBox_7)
        self.pushButton_9.setGeometry(QtCore.QRect(140, 80, 100, 30))
        self.pushButton_9.setObjectName(_fromUtf8("pushButton_9"))
        self.label_10 = QtGui.QLabel(self.groupBox_7)
        self.label_10.setGeometry(QtCore.QRect(140, 30, 100, 30))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_10.setFont(font)
        self.label_10.setObjectName(_fromUtf8("label_10"))
        self.lineEdit_3 = QtGui.QLineEdit(self.groupBox_7)
        self.lineEdit_3.setGeometry(QtCore.QRect(250, 30, 100, 30))
        self.lineEdit_3.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.lineEdit_3.setObjectName(_fromUtf8("lineEdit_3"))
        self.label_11 = QtGui.QLabel(self.groupBox_7)
        self.label_11.setGeometry(QtCore.QRect(360, 30, 20, 30))
        self.label_11.setObjectName(_fromUtf8("label_11"))
        self.groupBox_8 = QtGui.QGroupBox(self.tab_4)
        self.groupBox_8.setGeometry(QtCore.QRect(10, 140, 531, 341))
        self.groupBox_8.setObjectName(_fromUtf8("groupBox_8"))
        self.listWidget_3 = QtGui.QListWidget(self.groupBox_8)
        self.listWidget_3.setGeometry(QtCore.QRect(0, 20, 531, 321))
        self.listWidget_3.setObjectName(_fromUtf8("listWidget_3"))
        self.tabWidget.addTab(self.tab_4, _fromUtf8(""))
        CNN_Captcha.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(CNN_Captcha)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 580, 27))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menu = QtGui.QMenu(self.menubar)
        self.menu.setObjectName(_fromUtf8("menu"))
        CNN_Captcha.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(CNN_Captcha)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        CNN_Captcha.setStatusBar(self.statusbar)
        self.action = QtGui.QAction(CNN_Captcha)
        font = QtGui.QFont()
        self.action.setFont(font)
        self.action.setObjectName(_fromUtf8("action"))
        self.action_3 = QtGui.QAction(CNN_Captcha)
        self.action_3.setObjectName(_fromUtf8("action_3"))
        self.menu.addAction(self.action)
        self.menu.addSeparator()
        self.menu.addAction(self.action_3)
        self.menubar.addAction(self.menu.menuAction())

        self.retranslateUi(CNN_Captcha)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(CNN_Captcha)

    def retranslateUi(self, CNN_Captcha):
        CNN_Captcha.setWindowTitle(_translate("CNN_Captcha", "CNN_Captcha", None))
        self.groupBox.setTitle(_translate("CNN_Captcha", "输入图片", None))
        self.label.setText(_translate("CNN_Captcha", "本地图片：", None))
        self.pushButton.setText(_translate("CNN_Captcha", "浏览", None))
        self.label_2.setText(_translate("CNN_Captcha", "网络图片：", None))
        self.label_3.setText(_translate("CNN_Captcha", "识别结果：", None))
        self.label_4.setText(_translate("CNN_Captcha", "...", None))
        self.pushButton_3.setText(_translate("CNN_Captcha", "复制到剪贴板", None))
        self.pushButton_2.setText(_translate("CNN_Captcha", "开始识别", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("CNN_Captcha", "验证码识别", None))
        self.groupBox_3.setTitle(_translate("CNN_Captcha", "控制面板", None))
        self.pushButton_6.setText(_translate("CNN_Captcha", "停止训练", None))
        self.pushButton_4.setText(_translate("CNN_Captcha", "开始训练", None))
        self.pushButton_5.setText(_translate("CNN_Captcha", "暂停/继续", None))
        self.label_5.setText(_translate("CNN_Captcha", "训练次数：", None))
        self.comboBox.setItemText(0, _translate("CNN_Captcha", "   ∞", None))
        self.comboBox.setItemText(1, _translate("CNN_Captcha", " 3000", None))
        self.comboBox.setItemText(2, _translate("CNN_Captcha", " 5000", None))
        self.comboBox.setItemText(3, _translate("CNN_Captcha", " 7000", None))
        self.comboBox.setItemText(4, _translate("CNN_Captcha", " 10000", None))
        self.comboBox.setItemText(5, _translate("CNN_Captcha", " 30000", None))
        self.comboBox.setItemText(6, _translate("CNN_Captcha", " 50000", None))
        self.label_6.setText(_translate("CNN_Captcha", "模型保存频率：每", None))
        self.comboBox_2.setItemText(0, _translate("CNN_Captcha", "1000", None))
        self.comboBox_2.setItemText(1, _translate("CNN_Captcha", "500", None))
        self.comboBox_2.setItemText(2, _translate("CNN_Captcha", "3000", None))
        self.comboBox_2.setItemText(3, _translate("CNN_Captcha", "5000", None))
        self.comboBox_2.setItemText(4, _translate("CNN_Captcha", "10000", None))
        self.label_8.setText(_translate("CNN_Captcha", "次", None))
        self.groupBox_4.setTitle(_translate("CNN_Captcha", "训练日志", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("CNN_Captcha", "模型训练", None))
        self.groupBox_5.setTitle(_translate("CNN_Captcha", "控制面板", None))
        self.pushButton_8.setText(_translate("CNN_Captcha", "停止测试", None))
        self.pushButton_7.setText(_translate("CNN_Captcha", "开始测试", None))
        self.label_9.setText(_translate("CNN_Captcha", "测试次数：", None))
        self.comboBox_3.setItemText(0, _translate("CNN_Captcha", "  100", None))
        self.comboBox_3.setItemText(1, _translate("CNN_Captcha", "  500", None))
        self.comboBox_3.setItemText(2, _translate("CNN_Captcha", "  1000", None))
        self.comboBox_3.setItemText(3, _translate("CNN_Captcha", "  5000", None))
        self.comboBox_3.setItemText(4, _translate("CNN_Captcha", "  10000", None))
        self.groupBox_6.setTitle(_translate("CNN_Captcha", "测试日志", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("CNN_Captcha", "模型测试", None))
        self.groupBox_7.setTitle(_translate("CNN_Captcha", "控制面板", None))
        self.pushButton_10.setText(_translate("CNN_Captcha", "停止生成", None))
        self.pushButton_9.setText(_translate("CNN_Captcha", "开始生成", None))
        self.label_10.setText(_translate("CNN_Captcha", "验证码数量：", None))
        self.label_11.setText(_translate("CNN_Captcha", "个", None))
        self.groupBox_8.setTitle(_translate("CNN_Captcha", "生成日志", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("CNN_Captcha", "验证码生成", None))
        self.menu.setTitle(_translate("CNN_Captcha", "帮助", None))
        self.action.setText(_translate("CNN_Captcha", "设置", None))
        self.action_3.setText(_translate("CNN_Captcha", "关于", None))

