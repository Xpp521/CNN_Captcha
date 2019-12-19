import time
from PyQt4 import QtCore, QtGui

from cnn_sys import X, Y, keep_prob
from data_iter import get_next_batch
from cfg import save_model


class TrainThread(QtCore.QThread):
    signal = QtCore.pyqtSignal(QtCore.pyqtWrapperType, str, str)
    is_paused = None
    pause_flag = None
    status = None
    sess = None
    saver = None
    predict = None
    optimizer = None
    loss = None
    count = None
    rate = None

    def __init__(self):
        super(TrainThread, self).__init__()

    def run(self):
        self.signal.emit(QtGui.QListWidget, 'listWidget', '训练已开始。')
        step = 1
        while True:
            if not self.status:
                self.signal.emit(QtGui.QListWidget, 'listWidget', '训练已停止。')
                return
            if self.is_paused:
                if not self.pause_flag:
                    self.signal.emit(QtGui.QListWidget, 'listWidget', '训练已暂停。')
                    self.pause_flag = True
                time.sleep(0.1)
                continue
            batch_x, batch_y = get_next_batch(32)
            _, loss_ = self.sess.run([self.optimizer, self.loss], {X: batch_x, Y: batch_y, keep_prob: 0.98})
            # print(step, 'loss:\t', loss_)
            self.signal.emit(QtGui.QListWidget, 'listWidget', '%s loss:\t %s' % (step, loss_))
            if 0 == step % self.rate:
                self.saver.save(self.sess, save_model, step)
            if 0 == step % 100:
                batch_x_test, batch_y_test = get_next_batch(100)
                acc = self.sess.run(self.accuracy, {X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                # print(step, 'acc---------------------------------\t', acc)
                self.signal.emit(QtGui.QListWidget, 'listWidget',
                                 '%d acc---------------------------------\t%s' % (step, acc))
            if '   ∞' != self.count and int(self.count) == step:
                return
            step += 1
