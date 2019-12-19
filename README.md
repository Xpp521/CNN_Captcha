# 基于CNN算法的验证码识别系统
### 1、简介
可针对字母扭曲干扰型验证码进行识别。模型使用TensorFlow构建；训练集使用captcha库生成；GUI界面采用Qt编写；模型准确率可到达98%。

### 2、关键技术
- PyQt4
- TensorFlow

### 3、系统截图
#### （1）验证码识别：
![Predict](https://raw.githubusercontent.com/Xpp521/GUI/master/img/CNN_Captcha_Predict.jpg)

#### （2）模型训练：
![Train_Model](https://raw.githubusercontent.com/Xpp521/GUI/master/img/CNN_Captcha_Train_Model.jpg)

#### （3）模型测试：
![Test_Model](https://raw.githubusercontent.com/Xpp521/GUI/master/img/CNN_Captcha_Test_Model.jpg)

### 4、引用声明
部分代码参考自下文：
> https://github.com/zhengwh/captcha-tensorflow

在其基础上进行了优化，并使用Qt编写了图形化界面，使训练和测试等过程更加直观、方便。