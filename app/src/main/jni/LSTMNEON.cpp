#include "LSTMNEON.h"

#include "omp.h"

#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include "neon_mathfun.h"
// using ncnn::Mat;
using namespace ncnn;

// help function in neon
// I = 1.f / (1.f + exp(-I));
inline void sigmoid(float32x4_t& i) {
    float32x4_t one = vdupq_n_f32(1.0);
    i = vnegq_f32(i);
    i = exp_ps(i);
    i = vaddq_f32(i, one);
    i = vrecpeq_f32(i);
}

Mat transpose(Mat& m) {
    Mat ret;
    ret.create_like(m);
    int h = m.h;
    int w = m.w;
    int c = m.c;
    ret.reshape(h, w, c);
    // ret.h = m.w
    for (int i=0; i<w; i++) {
        float* ptr = ret.row(i);
        for (int j=0; j<h; j++) {
            float* ori_ptr = m.row(j);
            ori_ptr += c * i;
            for (int ci=0; ci<c; ci++) {
                *ptr = *ori_ptr;
                ptr++;
                ori_ptr++;
            }
        }
    }
    return ret;
}

LSTMNEON::LSTMNEON() {
    one_blob_only = true;
    support_inplace = false;
}

static int lstm(const Mat& bottom_blob, Mat& top_blob, int reverse, const Mat& weight_xc, const Mat& bias_c, const Mat& weight_hc, const Option& opt)
{
    int size = bottom_blob.w;
    int T = bottom_blob.h;

    int num_output = top_blob.w;

    // initial hidden state
    Mat hidden(num_output, 4u, opt.workspace_allocator);
    if (hidden.empty())
        return -100;

    // internal cell state
    Mat cell(num_output, 4u, opt.workspace_allocator);
    if (cell.empty())
        return -100;

    // 4 x num_output
    Mat gates(4*num_output, T, (int)1, (size_t)4u, opt.workspace_allocator);
    if (gates.empty())
        return -100;

    hidden.fill(0.f);
    cell.fill(0.f);

    // GEMM to calculate gates
    // gates.fill(0.f);

    // float32_t* weight_xc_ptr;
    // float32_t* x_ptr; 
    // float32_t* g_ptr;

    // GEMM
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int r=0; r<T; r+=4) {
        // int thread_id = omp_get_thread_num();
        float32_t* x_ptr = (float32_t*)bottom_blob.row(r);
        float32_t* g_ptr = (float32_t*)gates.row(r);
        for (int c=0; c<4*num_output; c+=4) {
            float32_t* weight_xc_ptr = (float32_t*)weight_xc.row(c);
            // fill 0
            float32x4_t C0 = vmovq_n_f32(0.0);
            float32x4_t C1 = vmovq_n_f32(0.0);
            float32x4_t C2 = vmovq_n_f32(0.0);
            float32x4_t C3 = vmovq_n_f32(0.0);
            for (int k=0; k<size; k++) {
                // float32x4_t X = {*(x_ptr+k), *(x_ptr+size*1+k), *(x_ptr+size*2+k), *(x_ptr+size*3+k)}; 
                float32x4_t W_X = {*(weight_xc_ptr+k), *(weight_xc_ptr+size*1+k), *(weight_xc_ptr+size*2+k), *(weight_xc_ptr+size*3+k)};
                // C0 = C0 + W_X * scalar
                C0 = vmlaq_n_f32(C0, W_X, *(x_ptr+k));
                C1 = vmlaq_n_f32(C1, W_X, *(x_ptr+size*1+k));
                C2 = vmlaq_n_f32(C2, W_X, *(x_ptr+size*2+k));
                C3 = vmlaq_n_f32(C3, W_X, *(x_ptr+size*3+k));
            }
            // *(g_ptr + xxx) = C0
            vst1q_f32(g_ptr + (4*num_output) * 0 + c, C0);
            vst1q_f32(g_ptr + (4*num_output) * 1 + c, C1);
            vst1q_f32(g_ptr + (4*num_output) * 2 + c, C2);
            vst1q_f32(g_ptr + (4*num_output) * 3 + c, C3);
        }
    }


    // unroll
    for (int t=0; t<T; t++) {
        // clip hidden by continuation indicator
        // h_cont_{t-1} = cont_t * h_{t-1}
        // h_cont_{t-1} = h_{t-1} if cont_t == 1
        //                0       otherwise
        // calculate hidden
        // gate_input_t := W_hc * h_conted_{t-1} + W_xc * x_t + b_c

        int cont = t > 0;

        int ti = reverse ? T-1-t : t;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<num_output; q+=4) {
            // Set gate IFGO for q th output
            // float32_t* x = (float32_t*)bottom_blob.row(ti);

            // Bias
            float32_t* bias_c_I = (float32_t*)(bias_c.row(0) + q);
            float32_t* bias_c_F = (float32_t*)(bias_c.row(1) + q);
            float32_t* bias_c_O = (float32_t*)(bias_c.row(2) + q);
            float32_t* bias_c_G = (float32_t*)(bias_c.row(3) + q);

            float32_t* gates_data = (float32_t*)(gates.row(ti) + q);
            
            // directly save to output to avoid adding

            // c
            float32_t* c_ptr = (float32_t*)(cell.row(0) + q);
            
            // h
            float32_t* h_ptr = (float32_t*)(hidden.row(0) + q);

            float32x4_t I = vld1q_f32(bias_c_I);
            float32x4_t F = vld1q_f32(bias_c_F);
            float32x4_t O = vld1q_f32(bias_c_O);
            float32x4_t G = vld1q_f32(bias_c_G);

            float32x4_t gate_I = vld1q_f32((gates_data + num_output * 0));
            float32x4_t gate_F = vld1q_f32((gates_data + num_output * 1));
            float32x4_t gate_O = vld1q_f32((gates_data + num_output * 2));
            float32x4_t gate_G = vld1q_f32((gates_data + num_output * 3));

            I = vaddq_f32(I, gate_I);
            F = vaddq_f32(F, gate_F);
            O = vaddq_f32(O, gate_O);
            G = vaddq_f32(G, gate_G);

            // Hc
            float32_t* weight_hc_I = (float32_t*)weight_hc.row(q + num_output * 0);
            float32_t* weight_hc_F = (float32_t*)weight_hc.row(q + num_output * 1);
            float32_t* weight_hc_O = (float32_t*)weight_hc.row(q + num_output * 2);
            float32_t* weight_hc_G = (float32_t*)weight_hc.row(q + num_output * 3);

            for (int i=0; i<num_output; i++)
            {
                float32_t h_cont = cont ? (float32_t)hidden[i] : 0.f;

                float32x4_t W_I = {*weight_hc_I, *(weight_hc_I+num_output), *(weight_hc_I+2*num_output), *(weight_hc_I+3*num_output)};
                float32x4_t W_F = {*weight_hc_F, *(weight_hc_F+num_output), *(weight_hc_F+2*num_output), *(weight_hc_F+3*num_output)};
                float32x4_t W_O = {*weight_hc_O, *(weight_hc_O+num_output), *(weight_hc_O+2*num_output), *(weight_hc_O+3*num_output)};
                float32x4_t W_G = {*weight_hc_G, *(weight_hc_G+num_output), *(weight_hc_G+2*num_output), *(weight_hc_G+3*num_output)};

                weight_hc_I += 1;
                weight_hc_F += 1;
                weight_hc_O += 1;
                weight_hc_G += 1;

                I = vmlaq_n_f32(I, W_I, h_cont);
                F = vmlaq_n_f32(F, W_F, h_cont);
                O = vmlaq_n_f32(O, W_O, h_cont);
                G = vmlaq_n_f32(G, W_G, h_cont);
            }
            vst1q_f32((gates_data + num_output * 0), I);
            vst1q_f32((gates_data + num_output * 1), F);
            vst1q_f32((gates_data + num_output * 2), O);
            vst1q_f32((gates_data + num_output * 3), G);
        }

            // avoid one load and store
            // gates_data[0] = I;
            // gates_data[1] = F;
            // gates_data[2] = O;
            // gates_data[3] = G;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<num_output; q+=4) {
            float32_t* gates_data = (float32_t*)(gates.row(ti) + q);

            float32x4_t I = vld1q_f32(gates_data + num_output * 0);
            float32x4_t F = vld1q_f32(gates_data + num_output * 1);
            float32x4_t O = vld1q_f32(gates_data + num_output * 2);
            float32x4_t G = vld1q_f32(gates_data + num_output * 3);
            
            // c
            float32_t* c_ptr = (float32_t*)(cell.row(0) + q);
            
            // h
            float32_t* h_ptr = (float32_t*)(hidden.row(0) + q);
            // lstm unit
            // sigmoid(I)
            // sigmoid(F)
            // sigmoid(O)
            // tanh(G)
            // c_t := f_t .* c_{t-1} + i_t .* g_t
            // h_t := o_t .* tanh[c_t]
            sigmoid(I);
            sigmoid(F);
            F = vmulq_n_f32(F, cont);
            sigmoid(O);
            G = tanh_ps(G);

            // calculate C2
            float32x4_t C2 = vld1q_f32(c_ptr);
            C2 = vmulq_f32(C2, F);
            C2 = vmlaq_f32(C2, I, G);
            vst1q_f32(c_ptr, C2);


            // calculate H
            C2 = tanh_ps(C2);
            C2 = vmulq_f32(C2, O);
            vst1q_f32(h_ptr, C2);
            
        }
        float32_t* output_data = (float32_t*)top_blob.row(ti);
        float32_t* h_ptr = (float32_t*)hidden.row(0);
        memcpy(output_data, h_ptr, num_output * sizeof(float32_t));
        // no cell output here
    }

    return 0;
}

int LSTMNEON::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int T = bottom_blob.h;

    Mat bottom_blob_padding;
    if (T & 0b11) {
        T >>= 2;
        T <<= 2;
        T += 4;
        bottom_blob_padding = Mat(bottom_blob.w, T, bottom_blob.c, 4u, opt.blob_allocator);
        bottom_blob_padding.fill(0.0);
        memcpy(bottom_blob_padding.data, bottom_blob.data, bottom_blob.total() * bottom_blob.elemsize);
    } else {
        bottom_blob_padding = bottom_blob;
    }

    int num_directions = direction == 2 ? 2 : 1;

    top_blob.create(num_output * num_directions, T, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // forward
    if (direction == 0)
    {
        int ret = lstm(bottom_blob_padding, top_blob, 0, weight_xc_data.channel(0), bias_c_data.channel(0), weight_hc_data.channel(0), opt);
        if (ret != 0)
            return ret;
    }

    if (direction == 1)
    {
        int ret = lstm(bottom_blob_padding, top_blob, 1, weight_xc_data.channel(0), bias_c_data.channel(0), weight_hc_data.channel(0), opt);
        if (ret != 0)
            return ret;
    }

    if (direction == 2)
    {
        Mat top_blob_forward(num_output, T, 4u, opt.workspace_allocator);
        if (top_blob_forward.empty())
            return -100;

        Mat top_blob_reverse(num_output, T, 4u, opt.workspace_allocator);
        if (top_blob_reverse.empty())
            return -100;

        int ret0 = lstm(bottom_blob_padding, top_blob_forward, 0, weight_xc_data.channel(0), bias_c_data.channel(0), weight_hc_data.channel(0), opt);
        if (ret0 != 0)
            return ret0;

        int ret1 = lstm(bottom_blob_padding, top_blob_reverse, 1, weight_xc_data.channel(1), bias_c_data.channel(1), weight_hc_data.channel(1), opt);
        if (ret1 != 0)
            return ret1;

        // concat w
        for (int i=0; i<T; i++)
        {
            const float* pf = top_blob_forward.row(i);
            const float* pr = top_blob_reverse.row(i);
            float* ptr = top_blob.row(i);

            memcpy(ptr, pf, num_output * sizeof(float));
            memcpy(ptr + num_output, pr, num_output * sizeof(float));
        }
    }

    return 0;
}