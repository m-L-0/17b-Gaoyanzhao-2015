# 17b-Gaoyanzhao-2015
## 高燕昭的大作业
### Fashion-mnist  
1. 将数据集划分成训练集、验证集、测试集并存储成TFRecord文件  
2. 利用matplotlib等工具对TFRecord中的样本数据进行可视化  
3. KNN算法对图片进行分类并训练  
4. K-Means算法对图片进行聚类  
5. CNN算法对图片进行分类  


### Vehicle_License_Plate_Recognition
#### 作业要求
1. 将分类好的图片及其标签序号存入到TFRecord文件中。  
2. 读取TFRecord文件：数据解码，reshape(恢复数据形状)。shuffle_batch。然后还有归一化处理、色彩空间变化、转换为灰色图片等操作。  
3. 设计卷积神经网络结构并利用卷积神经网络对汉字和字母数字分别进行训练。  
4. 利用测试集对卷积神经网络进行检测，并得到识别正确率。  
  
 ** CNN对字母数字字符识别  
 
 - 训练集 75% 验证集 25%  
 - 神经网络是两个卷积池化层，两个全连接层，利用tanh激活函数  
 - 第一次是5 * 5卷积 2 * 2 做池化 第二次是 5 * 5卷积 3 * 3 做池化  
 - 正则化 dropout = 0.3
 
 - 验证集正确率 97.6%  
 - 测试集正确率 93.66%  


### Captcha  

1. 统计数据集  
![123](/home/yanzhao/lastwork/Captcha/images/123.png)  
![234](/home/yanzhao/lastwork/Captcha/images/234.png)  
2. 数据集划分  
训练集、验证集、测试集以8:1:1划分
3. 模型设计  
cnn,3个卷积池化层，dropout = 0.5 ,优化函数优化
4. 训练结果  
- 训练集 正确率 98.04%  
- 验证集 正确率 90.2%  
- 测试集 正确率 91.2%  

