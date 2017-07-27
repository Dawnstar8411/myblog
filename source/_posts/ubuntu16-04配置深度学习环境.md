---
title: ubuntu16.04配置深度学习环境
date: 2017-06-27 09:38:24
tags:
---

最近在实验室配置了几个深度学习的服务器，在此记录下配置过程，以祭奠踩过的无数的坑。

### 环境列表
- Ubuntu 16.04 (64-bit)
- NVIDIA Taitan X
- Matlab2014b
- Python2.7 and Python3.5
- CUDA 8.0
- cuDNN 5.1
- opencv 3.1.0
- caffe
- caffe2
- Theano
- Keras
- Tensorflow
- Pytorch
- jupyter notebook

### Ubuntu系统安装
- 选择了Ubuntu 16.04版本，这个一个提供５年长期支持的版本
- 目前的主板都支持UEFI 模式安装，所以在安装分区时，要有一个EFI分区(500M即可)，引导文件将安装在这个分区中，最终的引导盘也要选择这个分区所在的盘符。不同于之前的Legacy模式中的`/boot`分区
- 安装系统前，先进入BIOS将`secure boot`关闭。因为如果开启了安全模式的话，很多第三方软件没法安装，尤其是安装显卡驱动时，会导致安装失败。

### 更换软件源
- 在系统设置中找到 软件和更新
- 在 Ubuntu软件栏目，找到“下载自”下拉菜单
- 其他站点->选择最佳服务器，选择离自己近的服务器就行了，我选择的是清华的服务器，下载速度快了十几倍。


### Ubuntu系统基础命令
```
sudo apt-get update  # 更新软件包列表
sudo apt-get upgrade # 升级软件包
sudo apt-get install build-essential # 安装该软件包之后，编译c/c++所需要的软件包都会被安装。 包括  libc6-dev, gcc,g++,make, dpkg-dev
sudo apt-get install cmake
sudo apt-get install gfortran  # 
sudo apt-get install git
sudo apt-get install pkg-config # 当从源代码编译软件时，用来提供依赖库的信息的软件
sudo apt-get install software-properties-common
sudo apt-get install wget # 一个可以空网络上自动下载文件的自由工具
sudo apt-get autoclean # 将已经删除了的软件包的.deb安装文件从硬盘中删除掉
sudo apt-get clean　# 把已安装的软件包的.deb安装包也删除掉
sudo apt-get autoremove　# 删除为了满足其他软件包的依赖而安装的，但现在不需要的软件包
sudo apt-get remove ＃删除已安装的软件包，保留配置文件
sudo apt-get --purge remove #　删除已安装的包，不保留配置文件
sudo rm -rf /var/lib/apt/lists/* # 删除apt-get install 的所有软件状态包
```
### Nvidia 驱动安装
- 查看显卡模式
```
lspci | grep -i nvidia
```
- 查看显卡驱动是否安装成功
```
cat /proc/driver/nvidia/version
```
- 方式一：在 [Nvidia 官网](http://www.geforce.com/drivers) 搜索合适的驱动，下载并安装。文件类型为`.run`。（不建议使用此方法）
```
Ctrl+Alt+F1  # 进入字符界面，输入用户名和密码登录
sudo service lightdm stop
sudo sh XXX.run
sudo service lightdm start
```
- 方式二：使用apt-get 来安装驱动。在["Proprietary GPU Drivers" PPA](https://launchpad.net/~graphics-drivers/+archive/ubuntu/ppa) 可以找到驱动的相关信息。该网站上有推荐安装的版本。（不建议使用此方法）
```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-375
```
- 方式三：直接安装CUDA，CUDA会自动安装相应的显卡驱动。

### CUDA
- 在[Nvidia官网](https://developer.nvidia.com/cuda-toolkit) 下载CUDA 8 并安装
```
# .deb格式的文件将显卡驱动和cuda工具包都一起安装了
sudo dpkg -i cuda-repo-ubuntu1604-8-0-local_8.0.44-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda
# .run格式的文件，安装时有一个人机交互的界面，可以选择是否安装显卡驱动，CUDA安装路径等。适合于先安装特殊需求的驱动，然后单独安装CUDA工具包
sudo sh cuda_8.0.61_375.26_linux.run
```
- 将 CUDA 添加至环境变量
```
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```
- 确认是否安装成功
```
nvcc -V
```
- 重启计算机
```
sudo shutdown -r now
```
- 修改gcc版本
目前ubuntu16.04自带gcc是5.4版本，但是CUDA8.0不支持５以上的版本。有人把系统中的gcc降级了，其实没必要这么做，甚至有可能对其他库的安装有影响。这里直接修改`host_config.h`文件，把`if __GNUC__>5`改成`if __GNUC__>6`即可
```
cd /usr/local/cuda-8.0/include
sudo cp host_config.h host_config.h.bak
sudo gedit host_config.h
```

#### 进行测试
- 安装CUDA测试样例，进行编译
```
/usr/local/cuda/bin/cuda-install-samples-8.0.sh ~/cuda-samples # 复制样例文件夹到根目录下
cd ~/cuda-samples/NVIDIA*Samples
make -j $(($(nproc) + 1)) # (-j $(($(nproc) + 1))) 表示进行并行编译，并行参数取决于CPU内核的个数
```

- 运行 deviceQuery，确保检测到了显卡并通过了测试
```
bin/x86_64/linux/release/deviceQuery
```

### cuDNN
- NVIDIA CUDA Deep Neural Network library(cuDNN)是深度神经网络中GPU加速相关的库，实现了很多基本的结构，如前传后传，卷积层，pooling 层，归一化，激活函数等。首先，需要在[https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn) 注册并下载。这里选择下载**cuDNN v5.1 library for Linux**
- 解压缩并复制文件到相应位置
```
tar xvf cudnn*.tgz  
cd cuda/include   # 这里文件下载到了根目录下，需按照自身情况选择目录
sudo cp cudnn.h /usr/local/cuda/include/
cd ~/cuda/lib64
sudo cp lib* /usr/local/cuda/lib64/
cd /usr/local/cuda/lib64/
sudo rm -rf libcudnn.so libcudnn.so.5 # 删除原有动态文件
sudo ln -s libcudnn.so.5.0.5 libcudnn.so.5 # 生成软链接
sudo ln -s libcudnn.so.5 libcudnn.so # 生成软链接
sudo ldconfig #更新链接，不然编译caffe会出现找不到cudnn的文件路径错误！
sudo chmod a+r /usr/local/cuda/lib64/libcudnn* # 将文件设置为所有人皆可读取
```
### GPU相关验证
- 到目前位置，跟GPU相关的东西都安装好了，使用指令`nvidia-smi`，可查看GPU当前状态。或者通过指令`nvidia-settings`查看GPU的详细信息


### MATLAB
- 下载matlab iso镜像，这里是MATLAB2014b
- 打开终端，输入`sudo mkdir /mnt/temp`（建立临时文件夹存放加载后的iso文件）
- 输入`sudo mount -o loop /path/to/matlab.iso /mnt/temp`
- 创建安装matlab的文件夹 `sudo mkdir /usr/local/matlab`
- 开始安装 `sudo /mnt/temp/install`选择不联网安装，序列号在`/mnt/temp/crack`文件夹下的`FIK 2014b.txt`文件里。安装路径选择为`/usr/local/matlab`
- 选择安装 `license manager`， 否则安装好以后无法激活
- 选择不联网激活，选择`/mnt/temp/crack`文件夹下的`license.lic`文件
- 将Crack 文件夹下的`libmwservices.so`文件拷贝到`/usr/local/matlab/bin/glnxa64`目录下
- 将matlab路径添加至环境变量，之后启动MATLAB只需要从终端输入`matlab`即可
```
sudo gedit ~/.bashrc
# 在最后一行添加
export PATH="/usr/local/matlab/bin:$PATH"
source ~/.bashrc
```
- **Note:** 打开matlab时会告知程序崩溃，强制退出，引起这种错误的原因是ubuntu15.04及以上版本包含更新版本的libstdc++.so.6，而matlab使用的是较旧版本(version 6.0.17)。当matlab首先加载 `/usr/local/matalb/sys/os/glnxa64`中的libstdc++.so.6.0.17时，操作系统收到一个matlab引起的不兼容错误，从而引发启动崩溃。这个bug已在R2016b中修复。
解决方案：
可以强制使matlab加载由操作系统提供的更新版的libstdc++库，通过以下步骤完成：
```
cd /usr/local/matlab/sys/os/glnxa64
sudo mv libstdc++.so.6 libstdc++.so.6.bak
```

### Python
- Ubuntu16.04 默认安装了Python2.7和Python3.5, 两个版本可同时存在，默认使用Python2.7。因为Ubuntu底层很多功能用到了Python2.7,所以不能将其删除。
- 切换版本：可通过直接输入`python2.7`和`python3.5`来切换版本。因为`/usr/bin`中 `python`是一个快捷方式，其默认指向`python2.7`。
- 在sublime Text3 中默认有python2的编译器，可多设置一个python3的编译器，使用时灵活切换
```
# Tools->Build system->New build System, 新建一个sublime-build文件
# 在打开的文件中填写如下内容，填写想要的python编译器的路径
{
 "cmd": ["/usr/bin/python3", "-u", "$file"],
 "file_regex": "^[ ]*File \"(...*?)\", line ([0-9]*)",
 "selector": "source.python" 
 }
 # 保存文件名为 python3.sublime-build. 运行Python文件时，sublime会调用Python3的编译器
```

#### pip install 与 apt-get install
- pip是Python的包管理工具，主要用于安装PyPI上的软件包，可以替代easy_install工具。
```
sudo apt-get install python-pip # 安装pip
sudo apt-get install python3-pip #安装pip3
```
- `/usr/bin`中有 `pip,pip2,pip3`，使用`pip2`为python2.7 安装包，用`pip3`为python3.5安装包，至于使用`pip`为那个版本的python安装，要看`pip`指向的是`pip2`还是`pip3`
```
pip --version
>>> pip 9.0.1 from /usr/local/lib/python3.5/dist-packages (python 3.5)
```
- pip install 的源是 pyPI, apt-get 的源是ubuntu 仓库。对于python 来说，pyPI 的源要比Ubuntu的多，对于同一个包，pyPI可提供多版本指定下载，apt-get install安装的是最新的系统化的包，在系统内完全安装。对apt-get install 来说，包的名称可能是`python-<packgeName>`,`python3-<packgeName>`；对于pip来说，直接使用`<packgeName>`来安装即可，至于为Python2还是Python3安装，要看pip的具体版本。

### OpenBLAS
- OpenBLAS 是一个线性代数库，可选择安装。
```
git clone https://github.com/xianyi/OpenBLAS.git
cd OpenBLAS
make FC=gfortran -j $(($(nproc)+1))
sudo make PREFIX=/usr/local install
```
- 添加环境变量
```
echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
```

### Python包安装
- 后面安装的深度学习框架都有Python接口，在安装它们之前，需安装好各种依赖项，以及我们可能会用到的Python包
```
sudo apt-get update
sudo apt-get install -y --no-install-recommends build-essential python-dev libgoogle-glog-dev libprotobuf-dev protobuf-compiler  
sudo apt-get install -y --no-install-recommends libgflags-dev # ubuntu 16.04 安装caffe2需要
sudo pip install -U --pre pip setuptools wheel
sudo pip install -U --pre numpy scipy matplotlib scikit-learn scikit-image protobuf nose ipython h5py sympy pygments sphinx
```
### Tensorflow
- 选取GPU支持版本
```
sudo pip install tensorflow-gpu
sudo pip3 install tensorflow-gpu # 为Python3 安装
```
- 测试
在终端进性测试，如果安装无误，便不会出现任何警告或错误
```
python
>>> import tensorflow as tf
>>> exit()
```

### Theano

- 安装运算加速库
```
sudo apt install -y libopenblas-dev liblapack-dev libatlas-base-dev
```

- 用pip安装Theano。
```
sudo pip install -U --pre  theano 
```
- 配置theano文件
```
sudo gedit ~/.theanorc
```
- GPU 加速
```
# GPU 加速
[global]
openmp=False
device=gpu
floatX=float32
allow_input_downcast=True
[lib]
cnmem=0.8
[root]
root=/usr/local/cuda-8.0
[blas]
ldflags=-lopenblas
[nvcc]
fastmath=True
[cuda]
root=/usr/local/cuda-8.0
```
- CPU加速
```
[global]
openmp=True
device=cpu
floatX=float32
allow_input_downcast=True
[blas]
ldflags=-lopenblas
```
- 在Python中测试是否安装成功
```
python
>>> import theano
>>> exit()
```

### Keras
- Keras 是Theano 和 Tensorflow 的封装，代码简单，可作为黑盒使用，方便学习和快速搭建网络
```
sudo pip install -U --pre keras
```

- 修改默认keras后端
```
gedit ~/.keras/keras.json
```
- 在python中验证
```
>>import keras
```

### Opencv 3.1
- 在git上下载opencv 和 opencv-contrib，这样以后有什么更新直接同步，而不用重新去官网下载。下载完后注意检出3.1.0分支
```
cd ~
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 3.1.0
cd ~
git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout 3.1.0
```
- CMake Opencv 源码
```
cd ~/opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE\
-D CMAKE_INSTALL_PREFIX=/usr/local\
-D WITH_TBB=ON\
-D WITH_V4L=ON\
-D WITH_QT=ON\
-D WITH_OPENGL=ON\
-D INSTALL_PYTHON_EXAMPLES=ON\
-D BUILD_EXAMPLES=ON\
-D INSTALL_C_EXAMPLES=ON\
-D CUDA_NVCC_FLAGS="-D_FORCE_INLINES"\
-D BUILD_NEW_PYTHON_SUPPORT=ON\
-D OPENCV_EXTRA_MODULES_PATH=~/Github/opencv_contrib/modules .. # 这是opencv_contrib的支持文件路径
```
- 编译
```
make -j $(($(nproc) + 1))
```
- 安装
```
sudo make install
sudo /bin/bash -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'
sudo ldconfig 
```
- **Note：** opencv3.1.0 与CUDA8.0不兼容，会导致错误`modules/cudalegacy/src/graphcuts.cpp:120:54: error: 
‘NppiGraphcutState’ has not been declared`。解决办法：修改～/opencv/modules/cudalegacy/src/graphcuts.cpp文件内容
```
//# if !defined(HAVE_CUDA) || difined(CUDA_DISABLER)
# if !defined(HAVE_CUDA) || defined(CUDA_DISABLER) || (CUDA_VERSION>=8000)
```
- **Note:** 编译中间会有一个ICV的文件下载的很慢，或者会校验错误。在网上下载相应的文件，拷贝到`～/opencv/3rdparty/ippicv/downloads/linux-808b791a6eac9ed78d32a7666804320e`目录下即可

### Caffe（Python2.7版本）
- 安装之前需要先安装一些依赖项
```
 libfreetype6-dev  libpng12-dev libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler  libboost-all-dev libgflags-dev libgoogle-glog-dev liblmdb-dev g++ 
```
- 从github克隆
```
https://github.com/BVLC/caffe.git
cd caffe
cp Makefile.config.example Makefile.config
```
- 修改Makefile.config文件`gksu gedit Makefile.config`
	- **`USE_CUDNN := 1`**  使用cudnn
	- **`OPENCV_VERSION := 3`** 使用opencv3.1.0版本
	- **`CUDA_DIR := /usr/local/cuda`** CUDA路径
	- **`BLAS := open`** 使用openBLAS版本
	- **`BLAS_INCLUDE := /usr/local/include`** BLAS路径
    - **`BLAS_LIB := usr/local/lib`**
	- **`MATLAB_DIR := /usr/local/matlab`** matlab 路径
	- **`PYTHON_INCLUDE := /usr/include/python2.7 \`
		`/usr/lib/python2.7/dist-packages/numpy/core/include`** 
	- **`PYTHON_LIB := /usr/lib`**
	- **` WITH_PYTHON_LAYER := 1`**
	- **`INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial/`**
    - **`LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial/`**
	- **` USE_PKG_CONFIG := 1`**
	- **`BUILD_DIR := build`**
    - **`DISTRIBUTE_DIR := distribute`**
	- **`TEST_GPUID := 0`**
	- **`Q ?= @`**
- 安装需要的包，编译源码
```
sudo pip2 install -r python/requirements.txt
make all -j8
make test -j8
make runtest -j8
```
- 编译pycaffe，作为caffe的Python接口
```
make pycaffe -j8
```
- 把caffe中和python 相关的内容的路劲刚添加到python的编译路径中
```
echo `export PYTHONPATH=(path/to/caffe)/python:$PYTHONPATH` >> ~/.bashrc
source ~/.bashrc
```

#### 编译caffe时可能出现的问题
- 为hdf5相关文件创建新的链接
	找到`libhdf_serial.so.10.1.0`所在的文件夹，建立两个链接`sudo ln -s libhdf5_serial.so.10.1.0 libhdf5.so`和 `sudo ln -s libhdf5_serial_hl.so.10.0.2 libhdf5_hl.so`
- 找不到 hdf5.h , hdf5_hl.h
	使用命令`sudo find /-name hdf5.h`和`sudo find / -name hdf5_hl.h`找到相应路径之后，把路径加到Makefile.config中。在"INCLUDE_DIRS"中添加`/usr/include/hdf5/serial/`
- 找不到libhdf5.so
	使用命令`sudo find / -name libhdf5.so`，找到路径后，为Makefile.config 中的"LIBRARY_DIRS"添加`/usr/lib/x86_64-linux-gnu/hdf5/serial/`
- 由于gcc版本太新导致的问题：在Makefile中搜索并替换
```
NVCCFLAGS += -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS) 

NVCCFLAGS += -D_FORCE_INLINES -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS) 

```
- usr/bin/ld: 找不到 -lippicv
	在opencv 中找到libippicv.a文件，将其复制到`/usr/local/lib`目录下
- 在Python中验证
```
python
>>> import caffe
>>> exit()
```




### Caffe2（Python2.7版本）

- Clone & Build
```
git clone --recursive https://github.com/caffe2/caffe2.git
cd caffe2
make
cd build
sudo make install
python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
python -m caffe2.python.operator_test.relu_op_test
```
- 设置环境变量 ~/.bashrc
```
export PYTHONPATH=/usr/local:$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/home/usrname/caffe2/build
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

### Pytorch
- 在[Pytorch官网](http://pytorch.org/) 中选择相应的系统，安装方式，Python版本，CUDA版本后，会给出两行代码，将其复制到命令行直接就可安装Pytorch。我安装时的代码如下：
```
pip install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp27-none-linux_x86_64.whl 
pip install torchvision
```

### Jupyter notebook

Jupyter notebook 的前身是Ipython notebook，安装好后在浏览器运行，可以在一个界面内编写代码，调试代码，添加笔记。如果将其安装在服务器端，可在客户机上通过浏览器访问。

- 在Ubuntu上安装Jupyter notebook
```
pip install jupyter notebook
```
- 在~/.bashrc 中添加路径
```
export PATH=$PATH:~/.local/bin
```
- 启动Jupyter notebook
```
jupyter notebook
```
#### Jupyter notebook 远程服务器配置
- 生成密码，打开python 终端
```
from IPython.lib import passwd
passwd()
>>> Enter password:
>>> Verify password:
>>> 'sha1:0e422dfccef2:84cfbcb 
b3ef95872fb8e23be3999c123f862d856' # 将此密码复制下来
```

- 编辑文件`jupyter_notebook_config.py`，在`~./jupyter/`文件夹中
```
c.NotebookApp.ip ='*'
c.NotebookApp.password = u' 上面复制的密码'
c.NotebookApp.port = 9999 # 这里填写端口号
c.InteractiveShellApp.matplotlib = 'inline'
```
- 如果没有~/.jupyter文件夹，终端输入
```
jupyter notebook --generate-config
```
- 在服务器端运行jupyter notebook之后，在其他电脑的浏览器上输入服务器的IP地址和端口号（e.g., 102.168.0.23:9999），然后输入密码即可使用。如果在服务器端运行，输入`localhost:9999`
- 可设置jupyter notebook 在后台不间断运行（可选）
```
nohup jupyter notebook >/dev/null 2>&1 &
```
- 查看jupyter内核空间
```
jupyter-kernelspec list

```
- 为jupyter添加Python3内核
```
pip3 install ipykernel
sudo python3 -m ipykernel install --name python3 --display-name "Python3.5.2"
```








