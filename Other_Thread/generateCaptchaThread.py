from os.path import join
from PIL import Image
from PyQt4 import QtCore, QtGui

from utils import wrap_gen_captcha_text_and_image
from cfg import workspace


class GenerateCaptchaThread(QtCore.QThread):
    signal = QtCore.pyqtSignal(QtCore.pyqtWrapperType, str, str)
    status = None
    count = None

    def __init__(self):
        super(GenerateCaptchaThread, self).__init__()

    def run(self):
        for i in range(1, self.count + 1):
            if not self.status:
                return
            text, image = wrap_gen_captcha_text_and_image()
            img = Image.fromarray(image)
            image_name = '%s.png' % text
            image_path = join(workspace, image_name)
            img.save(image_path)
            self.signal.emit(QtGui.QListWidget, 'listWidget_3', '【%s】\t生成成功~(%d/%d)' % (text, i, self.count))
