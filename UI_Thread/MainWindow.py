import os
from io import BytesIO

import pyperclip
import requests
import tensorflow as tf
import numpy as np
from PIL import Image

from PyQt4 import QtGui, QtCore
from UI_Thread.MainWindow_UI import Ui_CNN_Captcha

from utils import convert2gray, hack_function
from cnn_sys import crack_captcha_cnn, Y
from cfg import MAX_CAPTCHA, CHAR_SET_LEN, model_path, home_root, tb_log_path, workspace

from Other_Thread.InitThread import InitThread
from Other_Thread.TrainThread import TrainThread
from Other_Thread.TestThread import TestThread
from Other_Thread.generateCaptchaThread import GenerateCaptchaThread


class UIAssist(QtCore.QThread):
    """
    UI辅助线程
    """
    signal = QtCore.pyqtSignal()

    def __init__(self):
        super(UIAssist, self).__init__()

    def run(self):
        self.signal.emit()


class MainWindow(QtGui.QMainWindow):
    """
    主线程/UI线程。
    """

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_CNN_Captcha()
        self.ui.setupUi(self)

        # 为控件绑定事件处理方法
        self.ui.tabWidget.currentChanged.connect(self.tabWidget_currentChanged)
        self.ui.pushButton.clicked.connect(self.pushButton_clicked)
        self.ui.pushButton_2.clicked.connect(self.pushButton_2_clicked)
        self.ui.pushButton_3.clicked.connect(self.pushButton_3_clicked)
        self.ui.pushButton_4.clicked.connect(self.pushButton_4_clicked)
        self.ui.pushButton_5.clicked.connect(self.pushButton_5_clicked)
        self.ui.pushButton_6.clicked.connect(self.pushButton_6_clicked)
        self.ui.pushButton_7.clicked.connect(self.pushButton_7_clicked)
        self.ui.pushButton_8.clicked.connect(self.pushButton_8_clicked)
        self.ui.pushButton_9.clicked.connect(self.pushButton_9_clicked)
        self.ui.pushButton_10.clicked.connect(self.pushButton_10_clicked)
        self.ui.lineEdit_2.textChanged.connect(self.lineEdit_2_textChanged)
        # self.output = crack_captcha_cnn()
        # self.initThread = InitThread()
        # self.initThread.ui_signal.connect(self.display_message)
        # self.initThread.var_signal.connect(self.init0)
        # self.initThread.output = self.output
        # self.initThread.start()

        # 初始化训练线程
        self.trainThread = TrainThread()
        self.trainThread.signal.connect(self.display_message)
        self.trainThread.finished.connect(self.trainThread_finished)

        # 初始化测试线程
        self.testThread = TestThread()
        self.testThread.signal.connect(self.display_message)
        self.testThread.finished.connect(self.testThread_finished)

        # 初始化生成验证码线程
        self.generateCaptchaThread = GenerateCaptchaThread()
        self.generateCaptchaThread.signal.connect(self.display_message)
        self.generateCaptchaThread.finished.connect(self.generateCaptchaThread_finished)

        self.tab_flag = True  # 记录是否是首次切换到第2个标签
        self.tab = self.ui.tabWidget.currentIndex()  # 记录当前标签的id
        self.ui.tabWidget.setEnabled(False)  # 禁用标签控件

        # 初始化UI辅助线程
        self.uiAssist = UIAssist()
        self.uiAssist.signal.connect(self.init)
        self.uiAssist.start()

    # def init0(self, sess, predict, saver):
    #     self.sess = sess
    #     self.predict = predict
    #     self.saver = saver
    #     self.ui.tabWidget.setEnabled(True)

    def init(self):
        """
        初始化文件夹、tensorflow Session等。
        """
        if not os.path.exists(home_root):
            os.mkdir(home_root)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        if not os.path.exists(workspace):
            os.mkdir(workspace)
        if not os.path.exists(tb_log_path):
            os.mkdir(tb_log_path)
        self.ui.statusbar.showMessage('正在初始化tensorflow会话...')
        self.output = crack_captcha_cnn()
        self.saver = tf.train.Saver()
        self.trainThread.saver = self.saver
        for file in os.listdir(model_path):
            if 'checkpoint' == file:  # 模型文件存在
                self.predict = tf.argmax(tf.reshape(self.output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
                self.sess = tf.Session()
                self.saver.restore(self.sess, tf.train.latest_checkpoint(model_path))
                break
        else:  # 模型文件不存在
            self.ui.tab.setEnabled(False)  # 禁用验证码识别标签
            self.ui.tab_3.setEnabled(False)  # 禁用模型测试标签
            self.ui.tabWidget.setCurrentIndex(1)  # 显示模型训练标签
            self.predict = tf.reshape(self.output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
            self.sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
            self.sess.run(tf.global_variables_initializer())
        self.ui.tabWidget.setEnabled(True)
        self.ui.statusbar.showMessage('初始化完毕。')

    def display_message(self, widget_type, name, message):
        """
        展示信息方法。
        :param widget_type: 控件类型。
        :param name: 控件名称。
        :param message: 信息。
        """
        widget = self.findChild(widget_type, name)
        if QtGui.QStatusBar == widget_type:
            widget.showMessage(message)
        elif QtGui.QListWidget == widget_type:
            widget.addItem(message)
            widget.scrollToBottom()

    # def init(self,output, sess, predict, saver):
    #     self.output = output
    #     self.sess = sess
    #     self.predict = predict
    #     self.saver = saver
    #     self.ui.tabWidget.setEnabled(True)

    def tabWidget_currentChanged(self, tab_id):
        """
        切换标签处理方法。
        :param tab_id:新标签的id
        :return:
        """
        self.ui.statusbar.showMessage('切换到第%d个标签' % (tab_id + 1))
        if 1 == tab_id and self.tab_flag:  # 首次切换到第2个标签
            self.tab_flag = False
            self.predict = tf.reshape(self.output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])

            label = tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
            max_idx_p = tf.argmax(self.predict, 2)
            max_idx_l = tf.argmax(label, 2)
            correct_pred = tf.equal(max_idx_p, max_idx_l)
            with tf.name_scope('my_monitor'):
                self.trainThread.loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=Y))
            tf.summary.scalar('my_loss', self.trainThread.loss)
            self.trainThread.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.trainThread.loss)
            with tf.name_scope('my_monitor'):
                self.trainThread.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar('my_accuracy', self.trainThread.accuracy)

            self.sess.run(tf.global_variables_initializer())
            self.saver.restore(self.sess, tf.train.latest_checkpoint(model_path))
        elif 1 == tab_id and not self.tab_flag:  # 切换到第2个标签
            self.predict = tf.reshape(self.output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
            self.sess.run(tf.global_variables_initializer())
            self.saver.restore(self.sess, tf.train.latest_checkpoint(model_path))
        elif 1 == self.tab:  # 从第2个标签切换到其它标签
            self.predict = tf.argmax(tf.reshape(self.output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
            self.saver.restore(self.sess, tf.train.latest_checkpoint(model_path))
        self.tab = tab_id

    def pushButton_clicked(self):
        """
        选择本地图片方法。
        :return:
        """
        filename = QtGui.QFileDialog.getOpenFileName(self, 'Choose captcha', filter='*.jpg;*.png;*.webp')
        if '' != filename:
            self.ui.lineEdit.setText(filename)
            self.ui.lineEdit_2.clear()
            self.ui.pushButton_2.setEnabled(True)

    def lineEdit_2_textChanged(self):
        if '' != self.ui.lineEdit_2.text():
            self.ui.lineEdit.clear()
            self.ui.pushButton_2.setEnabled(True)
        else:
            self.ui.pushButton_2.setEnabled(False)

    def pushButton_2_clicked(self):
        """
        识别方法。
        """
        self.ui.statusbar.clearMessage()
        if '' != self.ui.lineEdit.text():
            pixmap = QtGui.QPixmap(self.ui.lineEdit.text()).scaled(self.ui.label_7.width(), self.ui.label_7.height())
            self.ui.label_7.setPixmap(pixmap)
            predict_text = self.recognize(open(self.ui.lineEdit.text(), 'rb').read())
            if predict_text is not None:
                self.ui.label_4.setText(predict_text)
                self.ui.pushButton_3.setEnabled(True)
        else:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                                     'Chrome/55.0.2883.87 Safari/537.36'}
            try:
                response = requests.get(self.ui.lineEdit_2.text(), headers=headers, timeout=7)
            except requests.Timeout:
                self.ui.statusbar.showMessage('连接超时，请检查你的网络连接。')
                return
            except Exception as e:
                print(e)
                self.ui.statusbar.showMessage('请求出错，请检查你输入的网址是否合法。')
                return
            pixmap = QtGui.QPixmap()
            pixmap.loadFromData(response.content)
            pixmap = pixmap.scaled(self.ui.label_7.width(), self.ui.label_7.height())
            self.ui.label_7.setPixmap(pixmap)
            predict_text = self.recognize(response.content)
            if predict_text is not None:
                self.ui.label_4.setText(predict_text)
                self.ui.pushButton_3.setEnabled(True)

    def recognize(self, bytes):
        """
        对验证码图片进行识别。
        :param bytes: 图片的二进制数据
        :return: 识别结果字符串
        """
        try:
            bin_img = Image.open(BytesIO(bytes))
        except OSError:
            self.ui.statusbar.showMessage('输入的数据不是验证码！')
            return
        if (160, 60) != bin_img.size:
            bin_img = bin_img.resize((160, 60))
        img = np.array(bin_img)
        img = convert2gray(img)
        img = img.flatten() / 255
        return hack_function(self.sess, self.predict, img)

    def pushButton_3_clicked(self):
        """
        将识别结果复制到剪贴板。
        """
        pyperclip.copy(self.ui.label_4.text())
        self.ui.statusbar.showMessage('【%s】已复制到剪贴板' % self.ui.label_4.text())

    def pushButton_4_clicked(self):
        """
        开始训练模型方法。
        """
        if self.testThread.isRunning():
            self.ui.statusbar.showMessage('正在测试模型，无法进行训练。')
            return
        self.trainThread.count = self.ui.comboBox.currentText()  # 训练次数
        self.trainThread.rate = int(self.ui.comboBox_2.currentText())  # 模型保存频率
        self.trainThread.status = True
        self.trainThread.is_paused = False
        self.trainThread.sess = self.sess
        self.trainThread.predict = self.predict
        self.trainThread.start()  # 开始训练
        self.ui.pushButton_4.setEnabled(False)
        self.ui.pushButton_5.setEnabled(True)
        self.ui.pushButton_6.setEnabled(True)

    def pushButton_5_clicked(self):
        """
        暂停/继续训练模型方法。
        """
        self.trainThread.is_paused = not self.trainThread.is_paused
        self.trainThread.pause_flag = False

    def pushButton_6_clicked(self):
        """
        停止训练模型方法。
        """
        self.trainThread.status = False
        self.ui.pushButton_4.setEnabled(True)
        self.ui.pushButton_5.setEnabled(False)
        self.ui.pushButton_6.setEnabled(False)

    def trainThread_finished(self):
        """
        模型训练结束响应方法。
        """
        self.ui.pushButton_4.setEnabled(True)
        self.ui.pushButton_5.setEnabled(False)
        self.ui.pushButton_6.setEnabled(False)

    def pushButton_7_clicked(self):
        """
        开始测试模型方法。
        """
        if self.trainThread.isRunning():
            self.ui.statusbar.showMessage('正在训练模型，无法进行测试。')
            return
        self.testThread.sess = self.sess
        self.testThread.predict = self.predict
        self.testThread.count = int(self.ui.comboBox_3.currentText())
        self.testThread.status = True
        self.ui.listWidget_2.addItem('测试已开始。')
        self.testThread.start()  # 开始测试
        self.ui.pushButton_7.setEnabled(False)
        self.ui.pushButton_8.setEnabled(True)

    def pushButton_8_clicked(self):
        """
        停止测试模型方法。
        """
        self.testThread.status = False
        self.ui.pushButton_8.setEnabled(False)
        self.ui.pushButton_7.setEnabled(True)
        self.ui.listWidget_2.addItem('测试已停止。')

    def testThread_finished(self):
        """
        模型测试结束响应方法。
        """
        self.ui.pushButton_8.setEnabled(False)
        self.ui.pushButton_7.setEnabled(True)

    def pushButton_9_clicked(self):
        """
        开始生成验证码方法。
        """
        self.generateCaptchaThread.count = int(self.ui.lineEdit_3.text())
        self.generateCaptchaThread.status = True
        self.ui.listWidget_3.addItem('开始生成验证码...')
        self.generateCaptchaThread.start()
        self.ui.pushButton_9.setEnabled(False)
        self.ui.pushButton_10.setEnabled(True)

    def pushButton_10_clicked(self):
        """
        停止生成验证码方法。
        """
        self.generateCaptchaThread.status = False
        self.ui.pushButton_9.setEnabled(True)
        self.ui.pushButton_10.setEnabled(False)
        self.ui.listWidget_3.addItem('已停止生成验证码')

    def generateCaptchaThread_finished(self):
        """
        验证码生成结束响应方法。
        """
        self.ui.pushButton_9.setEnabled(True)
        self.ui.pushButton_10.setEnabled(False)
