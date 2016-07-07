#include "SetFcn.h"


typedef float (*pDistanceFu) (float, float);
typedef float (*pDecayFu) (float, float, float);

// Custom Guassian that falls off to 0 (default falls off to 0.6)
__device__ static float distanceFunction(float dist, float sigmaT) {
    //return -(dist/sigmaT)+1;
	return exp(-pow(dist, 2.f)/(0.25f*pow(sigmaT, 2.f)));
    //return 0.5;
    /*float value = -(dist/sigmaT)+1;
    if (value < 0)
        value = 0;
    else if (value > 1)
        value = 1;
    return value;*/
}

__device__ static float distanceDecay (float sigma0, float T, float lambda) {
	return std::floor(sigma0*exp(-T/lambda) + 0.5f);
}

__device__ pDistanceFu pOwn = distanceFunction; 
__device__ pDecayFu pOwn2 = distanceDecay;

void SetFcn(ANN::DistFunction *fcn) {
	pDistanceFu hOwn;
        pDecayFu hOwn2;
	cudaMemcpyFromSymbol(&hOwn, pOwn, sizeof(pDistanceFu) );
        cudaMemcpyFromSymbol(&hOwn2, pOwn2, sizeof(pDecayFu) );
	fcn->distance = hOwn;
        fcn->rad_decay = hOwn2;
}
