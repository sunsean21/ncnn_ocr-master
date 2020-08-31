LOCAL_PATH := $(call my-dir)

# change this folder path to yours
NCNN_INSTALL_PATH := ${LOCAL_PATH}/ncnn-android-vulkan-lib

include $(CLEAR_VARS)
LOCAL_MODULE := ncnn
LOCAL_SRC_FILES := $(NCNN_INSTALL_PATH)/$(TARGET_ARCH_ABI)/libncnn.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)

LOCAL_MODULE := ocrncnn
LOCAL_SRC_FILES := ocr.cpp LSTMNEON.cpp LSTMGEMM.cpp LSTMDEFAULT.cpp

LOCAL_C_INCLUDES := $(NCNN_INSTALL_PATH)/include $(NCNN_INSTALL_PATH)/include/ncnn

LOCAL_STATIC_LIBRARIES := ncnn

LOCAL_CFLAGS := -O3 -fvisibility=hidden -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math
LOCAL_CPPFLAGS := -O3 -fvisibility=hidden -fvisibility-inlines-hidden -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math
LOCAL_LDFLAGS += -Wl,--gc-sections

LOCAL_CFLAGS += -fopenmp -static-openmp
LOCAL_CPPFLAGS += -fopenmp -static-openmp
LOCAL_LDFLAGS += -fopenmp -static-openmp

LOCAL_LDLIBS := -lz -llog -ljnigraphics -lvulkan -landroid
LOCAL_ARM_NEON := true
include $(BUILD_SHARED_LIBRARY)
