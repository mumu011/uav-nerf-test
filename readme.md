����Ŀ����instant-ngp�Ļ������޸Ķ����ģ����������ͼ����Ϣ����ros֮���ͨ��

��Ŀ������
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

ros��Ŀ��ַ��
https://github.com/ClearmanChen/uav-nerf-sim.git

���У�
�����ؽ���```python scripts/main.py ```
��rosͨ�Ų�ʵʱ�ؽ���```python scripts/ros_main.py```

�����޸ģ��ؽ����֣�default.yaml rosͨѶ���֣�ros_communication_v2.py
