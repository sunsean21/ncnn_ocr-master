#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>
#include <algorithm>

// ncnn
#include "net.h"
#include "benchmark.h"


#include "datareader.h"
#include "modelbin.h"
#include "util.h"

#include "LSTMDEFAULT.h"
#include "LSTMNEON.h"
#include "LSTMGEMM.h"

#define __BENCHMARK__
#define __NEONLSTM__
// #define __GEMMLSTM__

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

static std::vector<std::string> alphabetChinese;
static ncnn::Net crnn;
static ncnn::Net onelayer;


static const std::string DUMMY = "dummy.param";
//static const int T = 128;
//static const int num_output = 128;
//static const int num_directions = 1;
//static const int weight_data_size = 131072;
static const int T = 128;
static const int num_output = 256;
static const int num_directions = 1;
static const int weight_data_size = 262144;
static const int size = weight_data_size / num_directions / num_output / 4;


#ifdef __GEMMLSTM__
static const std::string MODELNAME = "crnn_lite_lstm_v2_gemm.param"; // name in asset, with neon lstm
#else
    #ifdef __NEONLSTM__
    static const std::string MODELNAME = "crnn_lite_lstm_v2_neon.param"; // name in asset, with neon lstm
    #else
    static const std::string MODELNAME = "crnn_lite_lstm_v2.param"; // name in asset
    #endif
#endif

static const std::string BINNAME = "crnn_lite_lstm_v2.bin";
static const std::string KEYNAME = "keys.txt";
static const float mean_vals_pse_angle[3] = { 0.485 * 255, 0.456 * 255, 0.406 * 255};
static const float norm_vals_pse_angle[3] = { 1.0 / 0.229 / 255.0, 1.0 / 0.224 / 255.0, 1.0 /0.225 / 255.0};
static const float mean_vals_crnn[1] = { 127.5};
static const float norm_vals_crnn[1] = { 1.0 /127.5};
static int num_thread = 4;
static int shufflenetv2_target_w  = 196;
static int shufflenetv2_target_h  = 48;
static int crnn_h = 32;

DEFINE_LAYER_CREATOR(LSTMNEON)
DEFINE_LAYER_CREATOR(LSTMDEFAULT)
DEFINE_LAYER_CREATOR(LSTMGEMM)

static void split_string(std::vector<std::string>& strings, const std::string& str, const std::string& delimiter)
{

    std::string::size_type pos = 0;
    std::string::size_type prev = 0;
    while ((pos = str.find(delimiter, prev)) != std::string::npos)
    {
        strings.push_back(str.substr(prev, pos - prev));
        prev = pos + 1;
    }

    // To get the last substring (or only, if delimiter is not found)
    strings.push_back(str.substr(prev));

}

std::vector<std::string> crnn_decode(const ncnn::Mat score , std::vector<std::string>& alphabetChinese) {
    float *srcdata = (float* ) score.data;
    std::vector<std::string> str_res;
    int last_index = 0;  
    for (int i = 0; i < score.h;i++){
        int max_index = 0;
        
        float max_value = -1000;
        for (int j =0; j< score.w; j++){
            if (srcdata[ i * score.w + j ] > max_value){
                max_value = srcdata[i * score.w + j ];
                max_index = j;
            }
        }
        if (max_index >0 && (not (i>0 && max_index == last_index))  ){
//            std::cout <<  max_index - 1 << std::endl;
//            std::string temp_str =  utf8_substr2(alphabetChinese,max_index - 1,1)  ;
            str_res.push_back(alphabetChinese[max_index-1]);
        }
        last_index = max_index;
    }
    return str_res;
}

extern "C" {
    
JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "OCR", "JNI_OnLoad");

    ncnn::create_gpu_instance();

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "OCR", "JNI_OnUnload");

    ncnn::destroy_gpu_instance();
}


JNIEXPORT jboolean JNICALL Java_com_eecs598_ocrncnn_OcrNcnn_Init(JNIEnv* env, jobject thiz, jobject assetManager)
{
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = num_thread;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;

    // use vulkan compute
    if (ncnn::get_gpu_count() != 0)
        opt.use_vulkan_compute = true;

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    crnn.opt = opt;

    onelayer.opt = opt;

    // add neon support custom layer
    crnn.register_custom_layer("LSTMDEFAULT", LSTMDEFAULT_layer_creator);
    crnn.register_custom_layer("LSTMNEON", LSTMNEON_layer_creator);
    crnn.register_custom_layer("LSTMGEMM", LSTMGEMM_layer_creator);

    onelayer.register_custom_layer("LSTMDEFAULT", LSTMDEFAULT_layer_creator);
    onelayer.register_custom_layer("LSTMNEON", LSTMNEON_layer_creator);
    onelayer.register_custom_layer("LSTMGEMM", LSTMGEMM_layer_creator);

    {
        int ret = crnn.load_param(mgr, MODELNAME.c_str());
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "OCR", "load_param_bin failed %s", MODELNAME.c_str());
            return JNI_FALSE;
        }
    }

    {
        int ret = crnn.load_model(mgr, BINNAME.c_str());
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "OCR", "load_model failed");
            return JNI_FALSE;
        }
    }

    // init words
    {
        AAsset* asset = AAssetManager_open(mgr, KEYNAME.c_str(), AASSET_MODE_BUFFER);
        if (!asset)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "OCR", "open keyfile failed");
            return JNI_FALSE;
        }

        int len = AAsset_getLength(asset);

        std::string words_buffer;
        words_buffer.resize(len);
        int ret = AAsset_read(asset, (void*)words_buffer.data(), len);

        AAsset_close(asset);

        if (ret != len)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "OCR", "read keyfile failed");
            return JNI_FALSE;
        }

        split_string(alphabetChinese, words_buffer, "\n");
        __android_log_print(ANDROID_LOG_DEBUG, "OCR", "alphabetChinese: %d", (int)alphabetChinese.size());
    }
    __android_log_print(ANDROID_LOG_DEBUG, "OCR", "here");


    // init one layer test

    {
        int ret = onelayer.load_param(mgr, DUMMY.c_str());
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "ONELAYER", "load_param_bin failed %s", DUMMY.c_str());
            return JNI_FALSE;
        }
    }

    int total_pd_size = weight_data_size + num_output * 4 * num_directions + num_output * num_output * 4 * num_directions;

    float* data = new float(total_pd_size * 2);

    if (!data) {
        __android_log_print(ANDROID_LOG_DEBUG, "ONELAYER", "failed here!");
        return JNI_FALSE;
    }

//    RandomizeFloat(data, total_pd_size);
    const unsigned char * A = (unsigned char*) data;
    DataReaderFromMemory dr(A);
    __android_log_print(ANDROID_LOG_DEBUG, "OCR", "here2");

    {
        int ret = onelayer.load_model(dr);
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "ONELAYER", "load_model failed");
            return JNI_FALSE;
        }
    }

    __android_log_print(ANDROID_LOG_DEBUG, "OCR", "here4");

    delete(data);
    return JNI_TRUE;
}

JNIEXPORT jstring JNICALL Java_com_eecs598_ocrncnn_OcrNcnn_Detect(JNIEnv* env, jobject thiz, jobject bitmap, jboolean use_gpu, jint in_thread_number, jint in_iteration) {
    if (use_gpu == JNI_TRUE && ncnn::get_gpu_count() == 0) {
        return env->NewStringUTF("no vulkan capable gpu");
    }
    // set number of threads
    num_thread = (int)in_thread_number;
    crnn.opt.num_threads = num_thread;
    int iteration = (int)in_iteration;
    __android_log_print(ANDROID_LOG_DEBUG, "OCR", "use_gpu %d, #thread %d, #iteration %d", use_gpu, num_thread, iteration);


    // Read bitmap
    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    int width = info.width;
    int height = info.height;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return NULL;
    float scale = crnn_h * 1.0/ height;
    int crnn_w_target = int(width * scale);
    ncnn::Mat in = ncnn::Mat::from_android_bitmap_resize(env, bitmap, ncnn::Mat::PIXEL_GRAY, crnn_w_target, crnn_h);
    in.substract_mean_normalize(mean_vals_crnn, norm_vals_crnn);
    
    std::vector<double> times;

    #ifdef __BENCHMARK__
    std::vector<double> times_first_lstm;
    std::vector<double> times_second_lstm;
    #endif

    ncnn::Mat crnn_preds;
    for (int it=0; it<iteration; it++) {
        double start_time = ncnn::get_current_time();
        double prev_time = start_time;
        // extractor
        ncnn::Extractor ex = crnn.create_extractor();
        ex.set_vulkan_compute(use_gpu);
        ex.set_num_threads(num_thread);
        ex.input("input", in);
        // ex.extract("out", crnn_preds);
        #ifdef __BENCHMARK__
        ncnn::Mat blob107;
        ex.extract("107", blob107);
        double time_used = ncnn::get_current_time() - prev_time;
        __android_log_print(ANDROID_LOG_DEBUG, "OCR_BENCHMARK", "before first lstm %.2fms", time_used);
        __android_log_print(ANDROID_LOG_DEBUG, "OCR_BENCHMARK", "LSTM 1st layer input: (h)%d x (w)%d x (c)%d", blob107.h, blob107.w, blob107.c);
        prev_time = ncnn::get_current_time();
        #endif

        ncnn::Mat blob162;
        ex.extract("234", blob162);
        #ifdef __BENCHMARK__
        time_used = ncnn::get_current_time() - prev_time;
        __android_log_print(ANDROID_LOG_DEBUG, "OCR_BENCHMARK", "1st lstm %.2fms", time_used);
        __android_log_print(ANDROID_LOG_DEBUG, "OCR_BENCHMARK", "LSTM 1st layer output: (h)%d x (w)%d x (c)%d", blob162.h, blob162.w, blob162.c);
        times_first_lstm.push_back(time_used);
        prev_time = ncnn::get_current_time();
        #endif

        // batch fc
        ncnn::Mat blob182(256, blob162.h);
        for (int i=0; i<blob162.h; i++)
        {
            ncnn::Extractor ex_1 = crnn.create_extractor();
            ex_1.set_num_threads(num_thread);
            ncnn::Mat blob162_i = blob162.row_range(i, 1);
            ex_1.input("253", blob162_i);

            ncnn::Mat blob182_i;
            ex_1.extract("254", blob182_i);

            memcpy(blob182.row(i), blob182_i, 256 * sizeof(float));
        }

        #ifdef __BENCHMARK__
        time_used = ncnn::get_current_time() - prev_time;
        __android_log_print(ANDROID_LOG_DEBUG, "OCR_BENCHMARK", "before 2nd lstm %.2fms", time_used);
        prev_time = ncnn::get_current_time();
        #endif

        // lstm
        ncnn::Mat blob243;
        ex.input("260", blob182);
        ex.extract("387", blob243);

        #ifdef __BENCHMARK__
        time_used = ncnn::get_current_time() - prev_time;
        __android_log_print(ANDROID_LOG_DEBUG, "OCR_BENCHMARK", "2nd lstm %.2fms", time_used);
        __android_log_print(ANDROID_LOG_DEBUG, "OCR_BENCHMARK", "LSTM 2nd layer input: (h)%d x (w)%d x (c)%d", blob182.h, blob182.w, blob182.c);
        __android_log_print(ANDROID_LOG_DEBUG, "OCR_BENCHMARK", "LSTM 2nd layer output: (h)%d x (w)%d x (c)%d", blob243.h, blob243.w, blob243.c);
        times_second_lstm.push_back(time_used);
        prev_time = ncnn::get_current_time();
        #endif
        // batch fc
        // TODO: add another layer for this part
        ncnn::Mat blob263(5530, blob243.h);
        for (int i=0; i<blob243.h; i++)
        {
            ncnn::Extractor ex_2 = crnn.create_extractor();
            ex_2.set_num_threads(num_thread);
            ncnn::Mat blob243_i = blob243.row_range(i, 1);
            ex_2.input("406", blob243_i);

            ncnn::Mat blob263_i;
            ex_2.extract("407", blob263_i);

            memcpy(blob263.row(i), blob263_i, 5530 * sizeof(float));
        }

        #ifdef __BENCHMARK__
        time_used = ncnn::get_current_time() - prev_time;
        __android_log_print(ANDROID_LOG_DEBUG, "OCR_BENCHMARK", "after 2nd lstm %.2fms", time_used);
        prev_time = ncnn::get_current_time();
        #endif

        crnn_preds = blob263;


        double elasped = ncnn::get_current_time() - start_time;
        times.push_back(elasped);
    }
    sort(times.begin(), times.end());
    // sort time used for lstm
    #ifdef __BENCHMARK__
    sort(times_first_lstm.begin(), times_first_lstm.end());
    sort(times_second_lstm.begin(), times_second_lstm.end());
    #endif
    auto res_pre = crnn_decode(crnn_preds, alphabetChinese);
    std::string res;
    for (int i=0; i<res_pre.size(); i++) {
        res += res_pre[i];
    }
    res += "  90%: ";
    __android_log_print(ANDROID_LOG_DEBUG, "OCR", "result_size %d", (int)res_pre.size());
    __android_log_print(ANDROID_LOG_DEBUG, "OCR", "result %s", res.c_str());
    int taillatencyindex = (int)(times.size() * 0.9);
    __android_log_print(ANDROID_LOG_DEBUG, "OCR", "total: %d, tail: %d", (int)times.size(), taillatencyindex);
    __android_log_print(ANDROID_LOG_DEBUG, "OCR", "tail latency: %.2fms   detect", times[taillatencyindex]);
    #ifdef __BENCHMARK__
    __android_log_print(ANDROID_LOG_DEBUG, "OCR", "1st tail: %.2fms   detect", times_first_lstm[taillatencyindex]);
    __android_log_print(ANDROID_LOG_DEBUG, "OCR", "2nd tail: %.2fms   detect", times_second_lstm[taillatencyindex]);
    #endif
    res += std::to_string(times[taillatencyindex]);
    res += " ms";
    jstring result = env->NewStringUTF(res.c_str());

    return result;
}


JNIEXPORT jstring JNICALL Java_com_eecs598_ocrncnn_OcrNcnn_Test(JNIEnv* env, jobject thiz, jboolean use_gpu, jint in_thread_number, jint in_iteration) {
    if (use_gpu == JNI_TRUE && ncnn::get_gpu_count() == 0) {
        return env->NewStringUTF("no vulkan capable gpu");
    }
    // set number of threads
    num_thread = (int)in_thread_number;
    crnn.opt.num_threads = num_thread;
    int iteration = (int)in_iteration;
    __android_log_print(ANDROID_LOG_DEBUG, "ONELAYER", "use_gpu %d, #thread %d, #iteration %d", use_gpu, num_thread, iteration);
    
//    int T = 256;



    std::vector<double> times;
    for (int i=0; i<iteration; i++) {
        ncnn::Mat in(size, T, 1, 4UL);
        RandomizeFloat((float*)in.data, size*T);
        ncnn::Extractor ex = onelayer.create_extractor();
        ex.set_vulkan_compute(use_gpu);
        ex.set_num_threads(num_thread);
        ex.input("in", in);
        double start_time = ncnn::get_current_time();
        ncnn::Mat out;
        ex.extract("out", out);
        double timeused = ncnn::get_current_time() - start_time;
        times.push_back(timeused);
    }

    sort(times.begin(), times.end());
    std::string log_result;
    for (int i=0; i<times.size(); i++) {
        log_result += std::to_string(times[i]);
        log_result += " ";
    }

    __android_log_print(ANDROID_LOG_DEBUG, "ONELAYER", "%s", log_result.c_str());

    int taillatencyindex = (int)(times.size() * 0.9);
    __android_log_print(ANDROID_LOG_DEBUG, "ONELAYER", "tail latency: %.2fms   detect", times[taillatencyindex]);
    std::string res = "90: ";
    res += std::to_string(times[taillatencyindex]);
    res += " ms";
    return env->NewStringUTF(res.c_str());
}


}   // extern "C"