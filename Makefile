cc        := g++
name      := trt_pipeline.so
workdir   := workspace
srcdir    := src
objdir    := objs
stdcpp    := c++17
cuda_home := /usr/local/cuda-12
cuda_arch := 8.6
nvcc      := $(cuda_home)/bin/nvcc -ccbin=$(cc)


project_include_path := src
opencv_include_path  := /compile/__install/opencv490-gst/include/opencv4
trt_include_path     := /opt/nvidia/TensorRT-10.9.0.34/include
cuda_include_path    := $(cuda_home)/include
ffmpeg_include_path  := 
bytetrack_include_path := src/3rd/ByteTrack/include
eigen_include_path   := /compile/__install/eigen/include
plog_include_path    := src/3rd/plog
cuvid_include_path   := src/3rd/cuvid-include

gst_include_path    := /usr/include/gstreamer-1.0/ /usr/include/glib-2.0 /usr/lib/x86_64-linux-gnu/glib-2.0/include

python_include_path  := /usr/include/python3.10


include_paths        := $(project_include_path) \
						$(opencv_include_path) \
						$(trt_include_path) \
						$(cuda_include_path) \
						$(python_include_path) \
						$(bytetrack_include_path) \
						$(eigen_include_path) \
						$(plog_include_path) \
						$(cuvid_include_path) \
						$(gst_include_path)


opencv_library_path  := /compile/__install/opencv490-gst/lib
trt_library_path     := /opt/nvidia/TensorRT-10.9.0.34/lib
cuda_library_path    := $(cuda_home)/lib64/
python_library_path  := 

library_paths        := $(opencv_library_path) \
						$(trt_library_path) \
						$(cuda_library_path) \
						$(cuda_library_path) \
						$(python_library_path)

link_ffmpeg       := avcodec avcodec avformat swresample swscale avutil
link_opencv       := opencv_core opencv_imgproc opencv_videoio opencv_imgcodecs
link_trt          := nvinfer nvinfer_plugin nvonnxparser nvcuvid nvidia-encode
link_cuda         := cuda cublas cudart cudnn
link_sys          := yaml-cpp stdc++ dl
gstreamer         := gstreamer-1.0 gobject-2.0 glib-2.0 gmodule-2.0 gthread-2.0 gstapp-1.0 gstbase-1.0 gstcontroller-1.0 gstpbutils-1.0 gstvideo-1.0

link_librarys     := $(link_opencv) $(link_trt) $(link_cuda) $(link_ffmpeg) $(link_sys) $(gstreamer)


empty := 
library_path_export := $(subst $(empty) $(empty),:,$(library_paths))

run_paths     := $(foreach item,$(library_paths),-Wl,-rpath=$(item))
include_paths := $(foreach item,$(include_paths),-I$(item))
library_paths := $(foreach item,$(library_paths),-L$(item))
link_librarys := $(foreach item,$(link_librarys),-l$(item))

cpp_compile_flags := -std=$(stdcpp) -w -O2 -m64 -fPIC -fopenmp -pthread  $(include_paths)
cu_compile_flags  := -Xcompiler "$(cpp_compile_flags)"
link_flags        := -pthread -fopenmp -Wl,-rpath='$$ORIGIN' $(library_paths) $(link_librarys)

cpp_srcs := $(shell find $(srcdir) -name "*.cpp")
cpp_objs := $(cpp_srcs:.cpp=.cpp.o)
cpp_objs := $(cpp_objs:$(srcdir)/%=$(objdir)/%)
cpp_mk   := $(cpp_objs:.cpp.o=.cpp.mk)

cu_srcs := $(shell find $(srcdir) -name "*.cu")
cu_objs := $(cu_srcs:.cu=.cu.o)
cu_objs := $(cu_objs:$(srcdir)/%=$(objdir)/%)
cu_mk   := $(cu_objs:.cu.o=.cu.mk)

TRT_VERSION := 10

# 根据 TRT_VERSION 设置不同的编译选项
ifeq ($(TRT_VERSION), 8)
    CXXFLAGS = -DTRT8
    cpp_srcs := $(filter-out src/common/tensorrt.cpp, $(cpp_srcs))
    cpp_objs := $(filter-out objs/common/tensorrt.cpp.o, $(cpp_objs))
else
    CXXFLAGS = -DTRT10
    cpp_srcs := $(filter-out src/common/tensorrt8.cpp, $(cpp_srcs))
    cpp_objs := $(filter-out objs/common/tensorrt8.cpp.o, $(cpp_objs))
endif

pro_cpp_objs := $(filter-out objs/interface.cpp.o, $(cpp_objs))

ifneq ($(MAKECMDGOALS), clean)
include $(mks)
endif


$(name)   : $(workdir)/$(name)

all       : $(name)

run       : $(name)
	@cd $(workdir) && python3 test.py

pro       : $(workdir)/pro

runpro    : pro
	@export LD_LIBRARY_PATH=$(library_path_export)
	@cd $(workdir) && ./pro

$(workdir)/$(name) : $(cpp_objs) $(cu_objs)
	@echo Link $@
	@mkdir -p $(dir $@)
	@$(cc) -shared $^ -o $@ $(link_flags)

$(workdir)/pro : $(pro_cpp_objs) $(cu_objs)
	@echo Link $@
	@mkdir -p $(dir $@)
	@$(cc) $^ -o $@ $(link_flags)

$(objdir)/%.cpp.o : $(srcdir)/%.cpp
	@echo Compile CXX $<
	@mkdir -p $(dir $@)
	@$(cc) $(CXXFLAGS) -c $< -o $@ $(cpp_compile_flags)

$(objdir)/%.cu.o : $(srcdir)/%.cu
	@echo Compile CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) $(CXXFLAGS) -diag-suppress 611 -c $< -o $@ $(cu_compile_flags)

$(objdir)/%.cpp.mk : $(srcdir)/%.cpp
	@echo Compile depends C++ $<
	@mkdir -p $(dir $@)
	@$(cc) -M $< -MF $@ -MT $(@:.cpp.mk=.cpp.o) $(cpp_compile_flags)

$(objdir)/%.cu.mk : $(srcdir)/%.cu
	@echo Compile depends CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -M $< -MF $@ -MT $(@:.cu.mk=.cu.o) $(cu_compile_flags)


clean :
	@rm -rf $(objdir) $(workdir)/$(name) $(workdir)/pro $(workdir)/*.trtmodel $(workdir)/imgs

.PHONY : clean run $(name) runpro