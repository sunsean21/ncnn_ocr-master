#ifndef _LSTMNEON_H_
#define _LSTMNEON_H_

#include <arm_neon.h>
#include "LSTMDEFAULT.h"
using namespace ncnn;
class LSTMNEON : public LSTMDEFAULT {
public:
    LSTMNEON();

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
};


#endif // LAYER_LSTM_H
