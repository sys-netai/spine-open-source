# Spine
Spine is a performant DRL-based congestion control algorithm for the Internet with ultra-low inference overhead. For details, please check the paper [Spine](https://cse.hkust.edu.hk/~kaichen/papers/spine-conext22.pdf).

This is the inference implementation with pre-trained model. The training architecture will be released later.

## Prerequisites
1. Please install the customized kernel of Astraea. See details in [Astraea](https://github.com/xudongliao/astraea-open-source/blob/main/kernel/deb/README.md).
2. We recommend to use g++-11 to compile spine programs.
3. CMake 3.20 or above is required.

## Requirements
> First you need to update pip, otherwise you may encounter some errors.
```bash
python3 -m pip install --upgrade pip
```
Then install the following packages:
- numpy
- torch
- matplotlib
- scikit-build

For DI-engine, you should install from the source code.
```bash
cd third_party/DI-engine
pip3 install -e .
```

## Setup spine kernel module
> We use vanilla kernel module by default.
```bash
cd spine-kernel/src
make
sudo ./spine_kernel_load.sh
```
## Build spine python helper
```bash
# please install required packages before building
cd src
mkdir build
cd build
cmake ..
make -j
```

## Run spine
1. Run spine kernel helper (for cross-space communication between kernel and user space spine program)
```bash
cd spine-kernel/python/src
sudo python3 spine_eval.py -u $USER
```

2. Run spine server
```bash
cd src/build
./bin/server --port=12345
```

3. Run spine client
```bash
cd src/build
./bin/client_spine \
--port=12345 \
--ip=127.0.0.1 \
--cong=vanilla \
--pyhelper=../../python/eval/spine_infer.py \
--model=../../python/model/current/ckpt/model.tar \
--interval=10000 \
```

# Citation
If you find this work useful, please cite our paper:
```
@inproceedings{spine,
  title={Spine: An efficient DRL-based congestion control with ultra-low overhead},
  author={Tian, Han and Liao, Xudong and Zeng, Chaoliang and Zhang, Junxue and Chen, Kai},
  booktitle={Proceedings of the 18th International Conference on emerging Networking EXperiments and Technologies},
  pages={261--275},
  year={2022}
}
```
