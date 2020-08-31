#ifndef _UTIL_H
#define _UTIL_H

#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cfloat>
static float RandomFloat(float LO = -1.f, float HI = 1.f)
{
    float random = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
    return random;
}

static void RandomizeFloat(float* ptr, int size, float LO = -1.f, float HI = 1.f)
{
    srand (static_cast <unsigned> (time(0)));
    for(int i=0; i< size; i++)
    {
        ptr[i] = RandomFloat(LO, HI);
    }
    return;
}
#endif // _UTIL_H