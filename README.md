# AroundViewMonitor-China-Developer
## 0. 项目说明  
国内某主机厂AVM（360全景环视）Pipeline Demo开源。  
包含内容：相机标定、鱼眼相机去畸变、360全景拼接融合、3D视角、辅助视角、自标定等。  
提供了可以跑通整个标定、360全景拼接融合流程的png鱼眼图、相机参数、融合mask等。  
具体算法原理见个人知乎专栏：https://www.zhihu.com/column/c_1696190558530326528。涉及算法细节可在知乎平台进行咨询。  
**合作或寻求技术支持** +V：tjy792864625  
### 0.1 360全景环视拼接  
https://github.com/user-attachments/assets/8c6956d2-8180-4960-aebd-b2b1e8775f64
### 0.2 3D 视角  
https://github.com/user-attachments/assets/b5dc3f79-09e2-4b82-96b3-e0de65cb84cc
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

## 3. 输入  
### 3.1 鱼眼图  
![图片1](https://github.com/user-attachments/assets/0aae2cd7-1046-479c-bf56-ae19bb2c6ee4)  
### 3.2 融合mask
![图片2](https://github.com/user-attachments/assets/ae246ea5-3a60-4861-990c-7763bb252464)  

## 3. 输出
### 3.1 360全景融合
![bev](https://github.com/user-attachments/assets/ed942443-6e68-45a1-91e1-6bf99050e97d)   

### 3.2 辅助视角（车轮\广角\3D等）
![assist](https://github.com/user-attachments/assets/a925cfda-2225-4d48-a92f-809318509ac1)




