# Model-based Reinforcement Learning for the Navigation of a Mobile Robot with Macnum Wheels

This is the code for our paper "Model-based Reinforcement Learning for the Navigation of a Mobile Robot with Macnum Wheels"

# Dependency

- python>=3.7
- pytorch>=1.7.0
- torchvision>=0.8.1
- numpy>=1.19.5
- ROS=melodic
- pyOpenGL=3.1.5
- glfw=2.4.0

# Usage
Our project is based on ROS, please make sure you have ROS installed. We uses a real McNamee wheel mobile robot, not a simulation, so you may not have a way to run our program directly. Please ensure you have the relevant hardware.

## Hardware

- McNamee Wheel Chassis
- STM32 Mini * 2
- IMU JY901B
- 2D Laser CU429-5000

## Run

If you have the above conditions, please run ddqn.py

## Relevant Documents

- ddqn.py - main
- virtual.py - our virtual environment where we train a GPM
- scene.py - OpenGL generates map
- agv.py - OpenGL generates an agv in map
- scene.py - OpenGL generates observation
- SE.py - the definition our network

# Citation

If you find this idea useful in your research, please consider citing:

```
@article{
  title={Model-based Reinforcement Learning for the Navigation of a Mobile Robot with Macnum Wheels},
  author={Jiahui Xu, Jinhao Liu, Gang Liu, Zhiyong Lv, Song Gao},
  journal={},
  year={2022}
}
```
