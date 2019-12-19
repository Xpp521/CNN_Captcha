from PyQt4 import QtCore, QtGui

from utils import hack_function, convert2gray, wrap_gen_captcha_text_and_image


class TestThread(QtCore.QThread):
    signal = QtCore.pyqtSignal(QtCore.pyqtWrapperType, str, str)
    status = None
    sess = None
    predict = None
    count = None

    def __init__(self):
        super(TestThread, self).__init__()

    def run(self):
        right_cnt = 0
        for i in range(1, self.count + 1):
            if not self.status:
                return
            text, image = wrap_gen_captcha_text_and_image()
            image = convert2gray(image)
            image = image.flatten() / 255
            predict_text = hack_function(self.sess, self.predict, image)
            if text == predict_text:
                right_cnt += 1
            else:
                # print("===========({}/{})\n标记: {}\n预测: {}".format(i, task_cnt, text, predict_text))
                self.signal.emit(QtGui.QListWidget, 'listWidget_2',
                                 "===========({}/{})\n标记: {}\n预测: {}".format(i, self.count, text, predict_text))
                # print("标记: {}  预测: {}".format(text, predict_text))
        # print('task:', task_cnt, ' cost time:', (time.time() - stime), 's')
        # print('right/total-----', right_cnt, '/', task_cnt)
        self.signal.emit(QtGui.QListWidget, 'listWidget_2',
                         'right/total-----{}/{}({}%)'.format(right_cnt, self.count, right_cnt * 100 / self.count))
