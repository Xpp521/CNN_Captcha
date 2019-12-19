import os
import tensorflow as tf
from PyQt4 import QtCore, QtGui

from cfg import MAX_CAPTCHA, CHAR_SET_LEN, home_root, model_path, tb_log_path
from cnn_sys import crack_captcha_cnn


class InitThread(QtCore.QThread):
    """
    初始化辅助线程。
    """
    ui_signal = QtCore.pyqtSignal(QtCore.pyqtWrapperType, str, str)
    var_signal = QtCore.pyqtSignal(object, object, object, object)
    output = []

    def __init__(self):
        super(InitThread, self).__init__()

    def run(self):
        """
        初始化文件夹、tensorflow Session等。
        :return:
        """
        if not os.path.exists(home_root):
            os.mkdir(home_root)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        if not os.path.exists(tb_log_path):
            os.mkdir(tb_log_path)
        self.ui_signal.emit(QtGui.QStatusBar, 'statusbar', '正在初始化tensorflow会话...')
        # output = crack_captcha_cnn()
        saver = tf.train.Saver()
        for file in os.listdir(model_path):
            if 'checkpoint' == file:                  # 模型文件存在
                predict = tf.argmax(tf.reshape(self.output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
                sess = tf.Session()
                saver.restore(sess, tf.train.latest_checkpoint(model_path))
                break
        else:                                         # 模型文件不存在
            # self.ui.tab.setEnabled(False)             # 禁用验证码识别标签
            # self.ui.tab_3.setEnabled(False)           # 禁用模型测试标签
            # self.ui.tabWidget.setCurrentIndex(1)      # 显示模型训练标签
            predict = tf.reshape(self.output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
            sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
            sess.run(tf.global_variables_initializer())
        
        self.var_signal.emit(sess, predict, saver)
        self.ui_signal.emit(QtGui.QStatusBar, 'statusbar', '初始化完毕。')
