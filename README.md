# ncnn_ocr
ncnn_ocr is a simple profiling tool for machine learning on mobile devices with [ncnn](https://github.com/Tencent/ncnn) platform. It is developed and used for profiling for my EECS598 System for AI's project.

## Install
The default release of ncnn has been included in the project file. The important thing is to get Android SDK and Android NDK. I build the project with Android Studio. To compile the project, you can

* Install Android Studio from https://developer.android.com/studio/
* The Android SDK will come with Android Studio.
* Install Android NDK from https://developer.android.com/ndk/downloads/
* Set the ANDROID_HOME and ANDROID_NDK_HOME
```bash
export ANDROID_HOME=path_to_sdk
export ANDROID_NDK_HOME=path_to_ndk
```
* Import the porject to Android Studio, build and install to your phone.

## Usage
The OCR model is from https://github.com/ouyanghuiyu/chineseocr_lite , it can be used to convert image to characters, by first selcting the target image and then click `TEST-SELECTED`, the `#threads` and `#iteration` to run the inference can be specified, finally, the test result and 90% tail latency will be displayed. `TEST-LSTM` is used to test one layer LSTM, the default setting has iteration 128, with number of output 256 and size 256. The 90% tail latency will be displayed. 

![Screenshot](https://raw.githubusercontent.com/runyuz/ncnn_ocr/master/screenshot/screenshot.jpg?token=AILUNJYIDFIAYJYPHTWYGXC6WJG2O)

