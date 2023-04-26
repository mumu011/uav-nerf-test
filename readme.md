此项目是在instant-ngp的基础上修改而来的，增加了深度图的信息和与ros之间的通信

项目构建：
```sh
$ git clone --recursive https://github.com/mumu011/uav-nerf-test.git
$ cd uav-nerf-test
```

```sh
instant-ngp$ cmake . -B build
instant-ngp$ cmake --build build --config RelWithDebInfo -j
or
cmake -DCMAKE_BUILD_TYPE=Debug . -B build
cmake --build build -j
```

ros项目地址：
https://github.com/ClearmanChen/uav-nerf-sim.git

运行：
场景重建：```python scripts/main.py ```
与ros通信并实时重建：```python scripts/ros_main.py```

参数修改：重建部分：default.yaml ros通讯部分：ros_communication_v2.py
