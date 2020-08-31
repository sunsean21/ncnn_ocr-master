#include "LSTMGEMM.h"
#include <math.h>
#include "neon_mathfun.h"
using namespace ncnn;

LSTMGEMM::LSTMGEMM()
{
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

    // GEMM
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int r=0; r<T; r++) {
        // int thread_id = omp_get_thread_num();
        float* x_ptr = (float*)bottom_blob.row(r);
        float* g_ptr = (float*)gates.row(r);
        for (int c=0; c<4*num_output; c+=4) {
            float* weight_xc_ptr = (float*)weight_xc.row(c);
            g_ptr[c] = 0;
            g_ptr[c+1] = 0;
            g_ptr[c+2] = 0;
            g_ptr[c+3] = 0;
            for (int k=0; k<size; k++) {
                float X = x_ptr[k];
                g_ptr[c] += X * weight_xc_ptr[k];
                g_ptr[c+1] += X * weight_xc_ptr[k + size * 1];
                g_ptr[c+2] += X * weight_xc_ptr[k + size * 2];
                g_ptr[c+3] += X * weight_xc_ptr[k + size * 3];
            }
        }
    }

    // unroll
    for (int t=0; t<T; t++)
    {
        // clip hidden by continuation indicator
        // h_cont_{t-1} = cont_t * h_{t-1}
        // h_cont_{t-1} = h_{t-1} if cont_t == 1
        //                0       otherwise
        // calculate hidden
        // gate_input_t := W_hc * h_conted_{t-1} + W_xc * x_t + b_c
        int cont = t > 0;

        int ti = reverse ? T-1-t : t;

        // Set gate IFGO for q th output
        const float* x = bottom_blob.row(ti);
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<num_output; q++)
        {
            const float* bias_c_I = bias_c.row(0);
            const float* bias_c_F = bias_c.row(1);
            const float* bias_c_O = bias_c.row(2);
            const float* bias_c_G = bias_c.row(3);

            float* gates_data = gates.row(ti) + q;

            // gate I F O G
            const float* weight_hc_I = weight_hc.row(num_output * 0 + q);
            const float* weight_hc_F = weight_hc.row(num_output * 1 + q);
            const float* weight_hc_O = weight_hc.row(num_output * 2 + q);
            const float* weight_hc_G = weight_hc.row(num_output * 3 + q);

            float I = bias_c_I[q] + gates_data[0];
            float F = bias_c_F[q] + gates_data[num_output*1];
            float O = bias_c_O[q] + gates_data[num_output*2];
            float G = bias_c_G[q] + gates_data[num_output*3];


            for (int i=0; i<num_output; i++)
            {
                float h_cont = cont ? hidden[i] : 0.f;

                I += weight_hc_I[i] * h_cont;
                F += weight_hc_F[i] * h_cont;
                O += weight_hc_O[i] * h_cont;
                G += weight_hc_G[i] * h_cont;
            }

            gates_data[0] = I;
            gates_data[1] = F;
            gates_data[2] = O;
            gates_data[3] = G;
        }

        // lstm unit
        // sigmoid(I)
        // sigmoid(F)
        // sigmoid(O)
        // tanh(G)
        // c_t := f_t .* c_{t-1} + i_t .* g_t
        // h_t := o_t .* tanh[c_t]
        float* output_data = top_blob.row(ti);
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q=0; q<num_output; q++)
        {
            const float* gates_data = gates.row(ti) + q;

            float I = gates_data[0];
            float F = gates_data[1 * num_output];
            float O = gates_data[2 * num_output];
            float G = gates_data[3 * num_output];

            I = 1.f / (1.f + exp(-I));
            F = cont ? 1.f / (1.f + exp(-F)) : 0.f;
            O = 1.f / (1.f + exp(-O));
            G = tanh(G);

            float cell2 = F * cell[q] + I * G;
            float H = O * tanh(cell2);
            cell[q] = cell2;
            hidden[q] = H;
            output_data[q] = H;
        }

        // no cell output here
    }

    return 0;
}

int LSTMGEMM::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int T = bottom_blob.h;

    int num_directions = direction == 2 ? 2 : 1;

    top_blob.create(num_output * num_directions, T, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // forward
    if (direction == 0)
    {
        int ret = lstm(bottom_blob, top_blob, 0, weight_xc_data.channel(0), bias_c_data.channel(0), weight_hc_data.channel(0), opt);
        if (ret != 0)
            return ret;
    }

    if (direction == 1)
    {
        int ret = lstm(bottom_blob, top_blob, 1, weight_xc_data.channel(0), bias_c_data.channel(0), weight_hc_data.channel(0), opt);
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

        int ret0 = lstm(bottom_blob, top_blob_forward, 0, weight_xc_data.channel(0), bias_c_data.channel(0), weight_hc_data.channel(0), opt);
        if (ret0 != 0)
            return ret0;

        int ret1 = lstm(bottom_blob, top_blob_reverse, 1, weight_xc_data.channel(1), bias_c_data.channel(1), weight_hc_data.channel(1), opt);
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

