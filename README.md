# 基于CNN算法的验证码识别系统
### 1、简介
本程序可对字母扭曲干扰型验证码进行识别。训练集使用captcha库生成；模型准确率可达98%。

### 2、关键技术
- PyQt4
- TensorFlow

### 3、系统截图
#### （1）验证码识别：
![Predict](https://raw.githubusercontent.com/Xpp521/CNN_Captcha/master/ScreenShots/Predict.png)

#### （2）模型训练：
![Train_Model](https://raw.githubusercontent.com/Xpp521/CNN_Captcha/master/ScreenShots/Train.png)

#### （3）模型测试：
![Test_Model](https://raw.githubusercontent.com/Xpp521/CNN_Captcha/master/ScreenShots/Test.png)

### 4、引用声明
部分代码参考自下文：
> https://github.com/zhengwh/captcha-tensorflow

在其基础上进行了优化，并使用Qt编写了图形化界面，使训练和测试等过程更加直观、方便。