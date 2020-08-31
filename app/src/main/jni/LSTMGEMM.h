#ifndef _LSTMGEMM_H_
#define _LSTMGEMM_H_

#include <arm_neon.h>
// #include "layer.h"
#include "LSTMDEFAULT.h"
using namespace ncnn;
class LSTMGEMM : public LSTMDEFAULT{
public:
    LSTMGEMM();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};


#endif // LAYER_LSTM_H
