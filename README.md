# Spine

## Requirements
- numpy
- torch

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
1. Run spine kernel helper
```bash
cd spine-kernel/python/src
sudo python spine_eval.py -u $USER
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
