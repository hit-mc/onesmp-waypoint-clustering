# 1. OneSMP Waypoint Clustering

使用聚类算法找出较近的地狱门，形成“簇”。
在地狱交通中，一个簇里的地狱门共享同一个冰道停靠点，可以简化地狱交通的设计。

# 2. 使用

1. 安装`requirements.txt`里的依赖。
2. 将坐标点存放在`waypoints.csv`文件里，编码设置为`utf-8`，格式可以参考*2.2*。
3. 运行`main.py`。

## 2.2 `waypoints.csv`格式参考

```
name,x,z
珊瑚机,-4084,532
刷冰机,-2069,-1581
```