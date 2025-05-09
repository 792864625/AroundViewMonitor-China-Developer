# AroundViewMonitor-China-Developer（360全景环视）
## 0. 项目说明  
国内某主机厂AVM（360全景环视）Pipeline Demo开源。  
包含内容：相机标定、鱼眼相机去畸变、360全景拼接融合、3D视角、辅助视角、自标定等。  
提供了可以跑通整个标定、360全景拼接融合流程的png鱼眼图、相机参数、融合mask等。  
具体算法原理见个人知乎专栏（涉及算法细节可在知乎平台进行**技术咨询**）：[https://www.zhihu.com/column/c_1696190558530326528     ](https://www.zhihu.com/people/bu-shou-hui-120-bu-gai-ming)  
**合作或寻求技术支持** +V(通过知乎)  
### 0.1 360全景环视拼接  
https://github.com/user-attachments/assets/8c6956d2-8180-4960-aebd-b2b1e8775f64
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






