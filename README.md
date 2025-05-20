# TRT PIPELINE INFER
使用tensorrt加载模型推理视频流，设计了一个pipeline模式，每个pipeline由不同的节点组成，每个节点对应不同的操作。

## 环境配置
基础镜像
nvcr.io/nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

### ffmpeg 4
```shell
apt -y update

apt install -y \
libass-dev libfreetype6-dev \
libjpeg-dev libpng-dev libtheora-dev \
libvorbis-dev libx11-dev libxfixes-dev \
libxcb-shm0-dev libxcb-xfixes0-dev \
libxext-dev libxrender-dev

apt install libavcodec-dev libavformat-dev libswscale-dev libavutil-dev -y
apt install ffmpeg -y
```

### gstreamer
```shell
apt install \
libssl-dev \
libgles2-mesa-dev \
libgstreamer1.0-0 \
gstreamer1.0-tools \
gstreamer1.0-plugins-good \
gstreamer1.0-plugins-bad \
gstreamer1.0-plugins-ugly \
gstreamer1.0-libav \
gstreamer1.0-rtsp \
libgstreamer-plugins-base1.0-dev \
libgstrtspserver-1.0-0 \
libgstrtspserver-1.0-dev \
libjansson4 \
libyaml-cpp-dev \
libjsoncpp-dev \
protobuf-compiler -y
```

### opencv编译
```shell
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/workspace/compile/__install/opencv490 \
-D BUILD_TESTS=OFF \
-D BUILD_PERF_TESTS=OFF \
-D BUILD_EXAMPLES=OFF \
-D BUILD_opencv_apps=OFF \
-D BUILD_PNG=ON \
-D BUILD_JPEG=ON \
-D BUILD_TIFF=ON \
-D BUILD_WEBP=ON \
-D OpenJPEG=ON \
-D BUILD_OPENEXR=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_EXTRA_MODULES_PATH=/workspace/compile/opencv490/opencv_contrib-4.9.0/modules \
-D WITH_CUDA=OFF \
-D WITH_CUDNN=OFF \
-D BUILD_PROTOBUF=ON \
-D OPENCV_DNN_CUDA=OFF \
-D CUDA_FAST_MATH=ON \
-D WITH_CUBLAS=OFF \
-D WITH_GSTREAMER=ON \
-D WITH_FFMPEG=ON \
-D WITH_QT=OFF \
-D WITH_GTK=OFF \
-D BUILD_JAVA=OFF \
-D WITH_1394=OFF ..

# 需要能找到 ffmpeg 和 gstreamer
--   Video I/O:
--     FFMPEG:                      YES
--       avcodec:                   YES (60.31.102)
--       avformat:                  YES (60.16.100)
--       avutil:                    YES (58.29.100)
--       swscale:                   YES (7.5.100)
--       avresample:                NO
--     GStreamer:                   YES (1.24.2)
--     v4l/v4l2:                    YES (linux/videodev2.h)

make -j$(nproc)
make install -j$(nproc)
```

## boost 安装
用于电子围栏计算多边形相交面积
```shell
apt-get install libboost-dev
```

## eigen 编译
```shell
cmake .. \
-D CMAKE_INSTALL_PREFIX=/compile/__install/eigen \
-D INCLUDE_INSTALL_DIR=include

make install
```

## TensorRT 安装
```shell
tar -zxvf TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-12.8.tar.gz -C /opt/nvidia/
```

## deepstream 安装
```shell
tar -xvf deepstream_sdk_v6.3.0_x86_64.tbz2 -C /
cd /opt/nvidia/deepstream/deepstream-6.3/
./install.sh
ldconfig
```

## nvcuvid
从宿主机拷贝到容器内
``` shell
docker cp /usr/lib/x86_64-linux-gnu/libnvcuvid.so.xxx.xxx container:/usr/lib/x86_64-linux-gnu/
docker cp /usr/lib/x86_64-linux-gnu/libnvidia-encode.so.xxx.xxx container:/usr/lib/x86_64-linux-gnu/
```

## 编译
根据自己的编译环境修改Makefile
```shell
make pro
```


## 节点介绍
### SOURCE 节点
能够从实时视频流、文件夹、本地视频中获取图片并加载到队列中。
- 硬解码
- 软解码
- 读取文件夹

---

### INFER 节点
使用tensorrt进行模型推理，获取模型识别结果。
- YOLOv8\YOLO11模型 关键点、检测、OBB、分割以及对应的sahi后模型推理
- YOLOV5模型 检测以及对应的sahi后模型推理

---

### TRACKER 节点
使用bytetrack对模型识别的结果进行目标跟踪

---

### 路由节点
多个pipeline共享同一个INFER节点后将不同pipeline的数据推送到对应的队列。
根据路由节点实现多条pipeline共用一个INFER节点, ROUTER节点会将识别后的数据根据pipeline id送入到对应的pipeline
```
N --- 1 --- N 结构
src1 ----->                       -----> osd1 -----> record1
              infer -----> router 
src2 ----->                       -----> osd2 -----> record2
```

---

### 画图节点
将模型识别和跟踪结果展示到图片上。
- 目标检测框及文字
- 目标检测旋转框及文字
- 实例分割Mask叠加
- 姿态估计关键点
- 目标追踪轨迹及ID

---

### ANALYZE 节点
分析模型识别的结果，用于不同任务。

---

### RECORDER 节点
将带有模型识别结果的图片输出到rtsp流、rtmp流或者本地文件。通过opencv gstreamer后端实现。
- 保存到本地命令
```c++
"appsrc ! queue ! videoconvert ! queue ! video/x-raw,format=I420 ! x264enc speed-preset=ultrafast bitrate=4000 "
"tune=zerolatency key-int-max=50 ! queue ! video/x-h264,profile=baseline ! queue ! "
"mp4mux ! filesink location=result/output_video.mp4";
```

- 软编码推送命令
```c++
"appsrc ! queue ! videoconvert ! queue ! video/x-raw,format=I420 ! x264enc speed-preset=ultrafast bitrate=4000 tune=zerolatency key-int-max=50 ! queue ! video/x-h264,profile=baseline ! queue ! rtspclientsink location=rtsp://172.16.20.168:8554/live1"
```

- 硬编码推送命令 （需要安装deepstream）
```c++
"appsrc ! queue ! video/x-raw,format=BGR ! nvvideoconvert gpu-id=1 ! video/x-raw(memory:NVMM),format=NV12 ! queue ! nvv4l2h264enc gpu-id=1 bitrate=4000000 control-rate=constant_bitrate iframeinterval=40 ! h264parse ! video/x-h264,profile=baseline ! queue ! rtspclientsink location=rtsp://172.16.20.168:8554/live1"
```

---

## PIPELINE
通过yaml配置文件或者代码组合不同的节点进行拉流识别
```yaml
nodes:
  - name: src1 # 节点的唯一名称
    type: STREAM
    max_pop_batch_size: 4
    config:
      gpu_id: 0
      decode_type: GPU
      skip_frame: 1
      stream_url: "xxx"
      stream_name: "src1"
      owner_pipeline_id: "1"
  - name: src2
    type: STREAM
    max_pop_batch_size: 4
    config:
      gpu_id: 0
      decode_type: GPU
      skip_frame: 1
      stream_url: "xxx"
      stream_name: "src2"
      owner_pipeline_id: "2"
  - name: pose
    type: INFER
    max_pop_batch_size: 1
    config:
      model_path: "model/engine/yolo11l-pose.transd.engine"
      names_path: "model/names/coco.names"
      max_batch_size: 16
      gpu_id: 0
      model_type: "YOLO11POSESAHI"
      conf_threshold: 0.5
      nms_threshold: 0.45
      auto_slice: false
      slice_width: 1000
      slice_height: 1000
      slice_horizontal_ratio: 0.6
      slice_vertical_ratio: 0.6
  - name: segment
    type: INFER
    max_pop_batch_size: 1
    config:
      model_path: "model/engine/yolo11l-seg.transd.engine"
      names_path: "model/names/coco.names"
      max_batch_size: 16
      gpu_id: 0
      model_type: "YOLO11SEGSAHI"
      conf_threshold: 0.25
      nms_threshold: 0.45
      auto_slice: false
      slice_width: 1000
      slice_height: 1000
      slice_horizontal_ratio: 0.2
      slice_vertical_ratio: 0.2
  - name: router
    type: ROUTER
    max_pop_batch_size: 4
  - name: tracker1
    type: TRACKER
    max_pop_batch_size: 4
    config:
      frame_rate: 20
      track_buffer: 30
      track_label: "person"
  - name: tracker2
    type: TRACKER
    max_pop_batch_size: 4
    config:
      frame_rate: 20
      track_buffer: 30
      track_label: "person"
  - name: entered
    type: ANALYZE
    max_pop_batch_size: 4
    config:
      task_name: entered
      fences: 
        - [[1276, 218], [1931, 98], [2319, 344], [1915, 745], [1478, 639]]
        - [[132, 1104], [939, 693], [1793, 1067], [1546, 1432], [491, 1428]]
  - name: osd1
    type: OSD
    max_pop_batch_size: 4
    config:
      show_final_result: false
      show_original_result: true
  - name: osd2
    type: OSD
    max_pop_batch_size: 4
    config:
      show_final_result: true
      show_original_result: false
  - name: record1
    type: RECORD 
    max_pop_batch_size: 4
    config:
      - element: "appsrc"
        properties:
          name: "appsrc1"
      - element: "queue"
      - element: "videoconvert"
      - element: "queue"
      - element: "video/x-raw,format=I420"
      - element: "x264enc"
        properties:
          speed-preset: "ultrafast"
          bitrate: 8000
          tune: "zerolatency"
          key-int-max: 50
      - element: "queue"
      - element: "video/x-h264,profile=baseline"
      - element: "queue"
      - element: "rtspclientsink"
        properties:
          location: "rtsp://172.16.20.193:8554/live101" 
  - name: record2
    type: RECORD
    max_pop_batch_size: 4
    config:
      - element: "appsrc"
        properties:
          name: "appsrc2"
      - element: "queue"
      - element: "videoconvert"
      - element: "queue"
      - element: "video/x-raw,format=I420"
      - element: "x264enc"
        properties:
          speed-preset: "ultrafast"
          bitrate: 8000
          tune: "zerolatency"
          key-int-max: 50
      - element: "queue"
      - element: "video/x-h264,profile=baseline"
      - element: "queue"
      - element: "rtspclientsink"
        properties:
          location: "rtsp://172.16.20.193:8554/live801"

pipelines:
  - id: "1"           # 流水线唯一ID (Unique pipeline ID)
    nodes:            # 该流水线包含的节点名称列表，按处理顺序排列 (List of node names in this pipeline, ordered by processing sequence)
      - src1         # 流水线起点 (Pipeline start)
      - segment         # 连接到共享的推理节点 (Connects to the shared inference node)
      - tracker1
      - osd1         # 连接到此流水线专属的 OSD 节点 (Connects to the OSD node specific to this pipeline)
      - record1      # 连接到此流水线专属的录制节点 (Connects to the Record node specific to this pipeline)
  - id: "2"           # 流水线唯一ID (Unique pipeline ID), 对应 C++ 中的 pipeline_id_2
    nodes:
      - src2
      - pose         # 两条流水线共享同一个 infer 节点实例 (Both pipelines share the same 'infer' node instance)
      - tracker2
      - entered
      - osd2
      - record2
```

## 后续任务
1. 动态删减pipeline
2. 动态修改配置文件
3. 做成http服务形式
