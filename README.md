# README: Installation Guide for Edge Caching Optimization

This repository contains code for dynamic adaptive caching strategies in edge computing, integrating LSTNetPro for user preference prediction and reinforcement learning (RL) for cache optimization.

## 1. User Preference Prediction Dependencies

To set up the environment for user preference prediction, install the following dependencies:

```python
# Install CUDA toolkit and cuDNN (compatible versions)
conda install cudatoolkit==11.3.1
conda install cudnn==8.3.1

# Install TensorFlow with GPU support
pip install tensorflow-gpu==2.7.0
pip install protobuf==3.20.3

# Install pandas for data processing
conda install pandas

# Install TCN (Temporal Convolutional Network) for LSTNetPro
pip install keras-tcn
pip install keras-tcn -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2. Cache Optimization Strategy Dependencies

For cache optimization using reinforcement learning, follow these installation steps:

```python
# Install CUDA toolkit and cuDNN (compatible versions)
conda install cudatoolkit==11.3.1
conda install cudnn==8.3.1

# Install TensorFlow with GPU support
pip install tensorflow-gpu==2.7.0
pip install protobuf==3.20.3

# Install pandas for data processing
conda install pandas

# Install TCN (Temporal Convolutional Network) for LSTNetPro
pip install keras-tcn
pip install keras-tcn -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Notes:

- Ensure you have an NVIDIA GPU compatible with CUDA 11.3 for optimal performance.
- Use `conda activate rl` before running reinforcement learning experiments.
- TensorFlow and PyTorch installations should match the CUDA version for compatibility.

This setup ensures smooth execution of the prediction model and reinforcement learning-based caching optimization.
