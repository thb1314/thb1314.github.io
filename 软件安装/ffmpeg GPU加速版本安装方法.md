---
comments: true
---


# FFMPEG Cuda版本编译安装方法

> 本文写于2023-06-23中午十二点

## 零、前言

最近在录制一些视频，需要上传到b站备份一下，上传上去后提示转码失败，遂采用ffmpeg将视频转码压制一下。

可怜我用了将近十年的笔记本电脑慢如龟速，就想着换上我配的2080TI主机，并采用显卡加速，编译完成后我准备将此过程记录下来，留给读者借鉴，因此就有了本篇文章。

## 一、ffmpeg介绍

官网：<http://ffmpeg.org/>

FFmpeg即是一款音视频编解码工具,同时也是一组音视频编码开发套件,作为编码开发套件,它为开发者提供了丰富的音视频处理的调用接口。

从使用者而不是而开发者角度来看，这个介绍足够了。

## 二、安装过程

FFMPEG的cuda版本需要编译安装，笔者采用的环境如下




```ini
OS:ubuntu22
gcc 7
cuda10.2
nvidia驱动版本  515.65.01
```




1. 确定安装目录结构




```shell
mkdir -p ~/ffmpeg/source # 存放源码
mkdir -p ~/ffmpeg/other_binary # 存放第三方库
mkdir -p ~/ffmpeg/binary # 存放ffmpeg可执行文件
```




最后的目录结构为



```shell
.
├── binary
├── other_binary
└── source
```




1. 在source目录下下载依赖库源码并编译

这里我省略切换到`cd source`的过程，读者自行添加




```shell
# 安装x264
git clone https://code.videolan.org/videolan/x264.git
./configure --prefix="$HOME/ffmpeg/other_binary/x264" --bindir="$HOME/ffmpeg/other_binary/x264/bin" --enable-shared --enable-pic  --disable-asm
make -j
make install

# 安装x265
git clone https://github.com/videolan/x265.git
cd x265
cd build/linux
cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$HOME/ffmpeg/other_binary/x265" -DENABLE_SHARED:bool=off .
make -j
make install

# 安装fdk-acc
sudo apt-get install -y libtool
git clone --depth 1 --branch v0.1.6 https://github.com/mstorsjo/fdk-aac.git
cd fdk-aac
autoreconf -fiv
./configure --prefix="$HOME/ffmpeg/other_binary/libfdk_aac"
make -j
make install

# 安装lame
curl -O -L http://downloads.sourceforge.net/project/lame/lame/3.99/lame-3.99.tar.gz
tar xzvf lame-3.99.tar.gz
cd lame-3.99
./configure --prefix="$HOME/ffmpeg/other_binary/libmp3lame" --bindir="$HOME/ffmpeg/other_binary/libmp3lame/bin"
make -j
make install

# 安装libopus
curl -O -L https://archive.mozilla.org/pub/opus/opus-1.2.1.tar.gz
tar xzvf opus-1.2.1.tar.gz
cd opus-1.2.1
./configure --prefix="$HOME/ffmpeg/other_binary/libopus"
make -j
make install

# 安装libogg
curl -O -L http://downloads.xiph.org/releases/ogg/libogg-1.3.3.tar.gz
tar xzvf libogg-1.3.3.tar.gz
cd libogg-1.3.3
./configure --prefix="$HOME/ffmpeg/other_binary/libogg"
make -j
make install

# 安装libvorbis
curl -O -L http://downloads.xiph.org/releases/vorbis/libvorbis-1.3.5.tar.gz
tar xzvf libvorbis-1.3.5.tar.gz
cd libvorbis-1.3.5
./configure --prefix="$HOME/ffmpeg/other_binary/libvorbis" --with-ogg="$HOME/ffmpeg/other_binary/libogg"
make -j
make install

# 安装libvpx
sudo apt-get install -y yasm
git clone --depth 1 -b v1.13.0 --single-branch https://github.com/webmproject/libvpx.git
cd libvpx
./configure --prefix="$HOME/ffmpeg/other_binary/libvpx" --disable-examples --disable-unit-tests --enable-vp9-highbitdepth --as=yasm --enable-shared
make -j
make install

# 安装nv-headers 根据自己的驱动库版本切换branch
git clone -b n11.1.5.2 --single-branch https://github.com/FFmpeg/nv-codec-headers.git
cd nv-codec-headers 
make
sudo make install
```




1. 安装ffmpeg支持cuda版本




```shell
# 安装ffmpeg
sudo apt install libfreetype6-dev
curl -O -L https://ffmpeg.org/releases/ffmpeg-6.0.tar.xz
tar xJvf ffmpeg-6.0.tar.xz
cd ffmpeg-6.0


# 安装
ffmpep_dep_include_dir=`find $HOME/ffmpeg/other_binary -name "include" -type d`
ffmpep_dep_lib_dir=`find $HOME/ffmpeg/other_binary -name "lib" -type d`
ffmpep_dep_package_dir=`find ~/ffmpeg -name pkgconfig`
ffmpep_dep_include_str=""
ffmpep_dep_lib_str=""
ffmpep_dep_package_str=""
for item in $ffmpep_dep_include_dir
do
    ffmpep_dep_include_str="$ffmpep_dep_include_str -I$item"
done
for item in $ffmpep_dep_lib_dir
do
    ffmpep_dep_lib_str="$ffmpep_dep_lib_str -L$item"
done
for item in $ffmpep_dep_package_dir
do
    ffmpep_dep_package_str="$item:$ffmpep_dep_package_str"
done


PKG_CONFIG_PATH="$PKG_CONFIG_PATH:$ffmpep_dep_package_str" ./configure \
  --prefix="$HOME/ffmpeg/binary/" \
  --pkg-config-flags="--static" \
  --extra-cflags="-I/usr/local/cuda/include $ffmpep_dep_include_str" \
  --extra-ldflags="-L/usr/local/cuda/lib64 $ffmpep_dep_lib_str" \
  --disable-static \
  --enable-shared \
  --extra-libs=-lpthread \
  --extra-libs=-lm \
  --enable-gpl \
  --enable-libfdk_aac \
  --enable-libfreetype \
  --enable-libmp3lame \
  --enable-libopus \
  --enable-libvorbis \
  --enable-libvpx \
  --enable-libx264 \
  --enable-libx265 \
  --enable-cuda-nvcc \
  --enable-cuda \
  --enable-libnpp \
  --disable-ffplay \
  --disable-doc \
  --enable-cuvid \
  --enable-nvenc \
  --enable-nonfree
make -j
make install
```



> 经验：
> 1. 缺什么工具就apt安装什么工具
> 2. 缺什么依赖库就编译安装该依赖库，实在不成功了再去apt install
> 3. 如果是环境问题，建议在docker环境下编译安装，然后把二进制文件以及相关依赖动态库copy出来即可

安装完成ffmpeg后直接打开是不能用的，原因是我们编译出来的大量的动态库并没有在系统动态库的搜索目录中，因此我们在`ff`写一个启动脚本。

`vim ~/ffmpeg/binary/bin/start_ffmpeg.sh`




```shell
#!/bin/bash


CURDIR=$(dirname $(realpath $0))
ffmpep_dep_lib_dir=`find $HOME/ffmpeg/other_binary -name "lib" -type d`
ffmpep_dep_package_str=""
for item in $ffmpep_dep_lib_dir
do
        ffmpep_dep_lib_str="$ffmpep_dep_lib_str:$item"
done
export LD_LIBRARY_PATH="$(dirname $CURDIR)/lib:$ffmpep_dep_lib_str:$LD_LIBRARY_PATH"
$CURDIR/ffmpeg $@
```




`chmod a+x ~/ffmpeg/binary/bin/start_ffmpeg.sh`

输入`~/ffmpeg/binary/bin/start_ffmpeg.sh -hwaccels`可以看到




```yaml
Hardware acceleration methods:
cuda
```



## 三、GPU加速转码命令

对于wmv3编码的wmv文件，我们可以采用如下命令将其转成wmv




```shell
~/ffmpeg/binary/bin/start_ffmpeg.sh -hwaccel cuda -extra_hw_frames 10 -c:v wmv3 -i input.wmv -c:v h264_nvenc -b:v 4000k -r 25 -preset slow output.mp4
```




* ffmpeg命令有一个特点是写在-i前面的算解码器参数，写在-i后面的算编码器参数
* `-r`：指定视频fps
* `-b:v`：指定视频比特率

我这里采用的原则是输出编码为`h264`，且采用适配的nvidia编码库。

输出视频的fps与码率与原视频相同，采用上述命令转码时，我的GPU算力利用率约为10%。

`speed`为8，即1s处理视频中8s的数据。

OK，就写到这吧，后续疑问欢迎在评论区交流。

