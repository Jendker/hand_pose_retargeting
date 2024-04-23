# Hand Pose Retargeting

## Introduction

This is the source code to the 2021 IEEE ICDL paper "Human hand motion retargeting for dexterous robotic hand".

You can find 2 branches here:
- main - has python 3 support. This updated code includes an implementation with Mujoco physics simulator and pose optimization with particle swarm optimization (PSO). Source code was developed in 2019 and I cannot guarantee compatibility with current version of Mujoco 3.x.
- python2.7_vrep - old version of the code which directly relates to the paper. It support VRep simulator (now CoppeliaSim). I cannot guarantee compatibility with the current version of CoppeliaSim.

If this work is helpful, please consider using following entry for the citation:

```
@inproceedings{orbikInverseReinforcementLearning2021,
  title = {Inverse Reinforcement Learning for Dexterous Hand Manipulation},
  booktitle = {2021 {{IEEE International Conference}} on {{Development}} and {{Learning}} ({{ICDL}})},
  author = {Orbik, Jedrzej and Agostini, Alejandro and Lee, Dongheui},
  year = {2021},
  month = aug,
  pages = {1--7},
  publisher = {IEEE},
  address = {Beijing, China},
  doi = {10.1109/ICDL49984.2021.9515637},
  urldate = {2023-02-25},
  isbn = {978-1-72816-242-3},
  langid = {english}
}
```

## ROS

This package works with Python 3. To use it with ROS Kinetic or newer:

```
cd ~/catkin_ws
rm -rf devel build
catkin build
source devel/setup.zsh  # or setup.bash
cd src 
git clone https://github.com/ros/geometry
git clone https://github.com/ros/geometry2
cd ..
rosdep install --from-paths src --ignore-src -y -r  # not sure if needed
catkin build --cmake-args \
            -DPYTHON_EXECUTABLE=/usr/bin/python3 \
            -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
            -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
source devel/setup.zsh  # or setup.bash
```
(source https://answers.ros.org/question/326226/importerror-dynamic-module-does-not-define-module-export-function-pyinit__tf2/ and https://github.com/ros/geometry2/issues/259)
