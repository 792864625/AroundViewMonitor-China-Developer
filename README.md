# AroundViewMonitor-China-Developer
## 0. 项目说明  
基于国内某主机厂AVM Pipeline开源的demo（360全景环视）包含内容：相机标定、鱼眼相机去畸变、360拼接融合、3D视角、辅助视角、自标定等。  
提供了可以跑通整个标定、360全景融合流程的鱼眼图、相机参数、融合mask等。
具体算法原理见个人知乎专栏：https://www.zhihu.com/column/c_1696190558530326528。  
合作+V：tjy792864625



## 1. 环境配置  
```
visual studio 2019
opencv 4.5.0
opencv_contrib-4.5.0
```

## 2. demo运行说明
```
2.1 计算鱼眼畸变系数
python correct.py
2.2 运行avm demo
test.cpp
```

## 3. 输入输出  
输入：鱼眼图  
![图片1](https://github.com/user-attachments/assets/0aae2cd7-1046-479c-bf56-ae19bb2c6ee4)  
输入：融合mask
![图片2](https://github.com/user-attachments/assets/ae246ea5-3a60-4861-990c-7763bb252464)  

输出：360全景融合, 各种辅助视角，3D视角等  
![bev](https://github.com/user-attachments/assets/ed942443-6e68-45a1-91e1-6bf99050e97d)  


