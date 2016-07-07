#ifndef _SOMKERNELS_
#define _SOMKERNELS_

#include "math/Random.h"
#include "math/Functions.h"
#include "Functors.h"
#include "SOMNetGPU.h"

#include <cfloat>
#include <cassert>
#include <cmath>
#include <algorithm>

#include <omp.h>

#include <thrust/extrema.h>
#include <thrust/distance.h>
#include <thrust/device_vector.h>

using namespace ANNGPGPU;


typedef float (*pDistanceFu) (float, float);
__device__ pDistanceFu pBubble 		= ANN::fcn_bubble_nhood; 
__device__ pDistanceFu pGaussian 	= ANN::fcn_gaussian_nhood; 
__device__ pDistanceFu pCutGauss 	= ANN::fcn_cutgaussian_nhood; 
__device__ pDistanceFu pMexican 	= ANN::fcn_mexican_nhood; 
__device__ pDistanceFu pEpanech 	= ANN::fcn_epanechicov_nhood;

bool SOMNetGPU::AssignDistanceFunction() {
	pDistanceFu hBubble; 
	pDistanceFu hGaussian; 
	pDistanceFu hCutGauss; 
	pDistanceFu hMexican; 
	pDistanceFu hEpanech;

	cudaMemcpyFromSymbol(&hBubble, pBubble, sizeof(pDistanceFu) );
	cudaMemcpyFromSymbol(&hGaussian, pGaussian, sizeof(pDistanceFu) );
	cudaMemcpyFromSymbol(&hCutGauss, pCutGauss, sizeof(pDistanceFu) );
	cudaMemcpyFromSymbol(&hMexican, pMexican, sizeof(pDistanceFu) );
	cudaMemcpyFromSymbol(&hEpanech, pEpanech, sizeof(pDistanceFu) );

	if (strcmp (GetDistFunction()->name, "gaussian") == 0) {
		GetDistFunction()->distance = hGaussian;
	} else if (strcmp (GetDistFunction()->name, "mexican") == 0) {
		GetDistFunction()->distance = hMexican;
	} else if (strcmp (GetDistFunction()->name, "bubble") == 0) {
		GetDistFunction()->distance = hBubble;
	} else if (strcmp (GetDistFunction()->name, "cutgaussian") == 0) {
		GetDistFunction()->distance = hCutGauss;
	} else if (strcmp (GetDistFunction()->name, "epanechicov") == 0) {
		GetDistFunction()->distance = hEpanech;
	} else {
		printf("No preimplemented function recognized. No assignment done.");
		return 0;
	}
	printf("Preimplemented function recognized. Assignment done.");
	return 1;
}

bool SOMNetGPU::DeassignDistanceFunction() {
	if (strcmp (GetDistFunction()->name, "gaussian") == 0) {
		GetDistFunction()->distance = ANN::fcn_gaussian_nhood; 
	} else if (strcmp (GetDistFunction()->name, "mexican") == 0) {
		GetDistFunction()->distance = ANN::fcn_mexican_nhood; 
	} else if (strcmp (GetDistFunction()->name, "bubble") == 0) {
		GetDistFunction()->distance = ANN::fcn_bubble_nhood;
	} else if (strcmp (GetDistFunction()->name, "cutgaussian") == 0) {
		GetDistFunction()->distance = ANN::fcn_cutgaussian_nhood;
	} else if (strcmp (GetDistFunction()->name, "epanechicov") == 0) {
		GetDistFunction()->distance = ANN::fcn_epanechicov_nhood;
	} else {
		printf("No preimplemented function recognized. No deassignment done.");
		return 0;
	}
	printf("Preimplemented function recognized. Deassignment done.");
	return 1;
}

// new reference implementation
ANNGPGPU::BMUExport hostGetMin(std::vector<ANNGPGPU::BMUExport> &vec) {
	assert(vec.size() > 0);
	if(vec.size() > 1) {
		std::sort(vec.begin(), vec.end() );
	}
	return *vec.begin();
}

// fast when maps are big
std::pair<float, unsigned int> devGetMin(const thrust::device_vector<float> &vec) {
	thrust::device_vector<float>::const_iterator d_min = thrust::min_element(vec.begin(), vec.end() );
	unsigned int iID = thrust::distance(vec.begin(), d_min);
	return std::pair<float, unsigned int>(*d_min, iID);
}

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
/*
 * Layout of SOMEdgeF2DArray:
 * 		COL1	COL2	COL3	COL(n+1)
 * ROW1		toNeur1	toNeur1	toNeur1	..
 * ROW2		toNeur2	toNeur2	toNeur2	..
 * ROW3		toNeur3	toNeur3	toNeur3	..
 * ROW(n+1)	..		..		..
 */
BMUExport
hostSOMFindBMNeuronID(std::vector<SOMExport*> &SExp,
		const float &fConscRate)
{
	BMUExport resBMU;
	std::vector<ANNGPGPU::BMUExport> vBMUExp(SExp.size() );

	assert(SExp.size() > 0);
	assert(vBMUExp.size() == SExp.size() );

	omp_set_num_threads(SExp.size() );  							// create as many CPU threads as there are CUDA devices
	#pragma omp parallel 									// for(int iDevID = 0; iDevID < static_cast<int>(SExp.size() ); iDevID++) {
	{
		unsigned int iDevID 	= omp_get_thread_num();
		checkCudaErrors(cudaSetDevice(iDevID) );
		
		unsigned int iWidth 	= SExp.at(iDevID)->f2dEdges.GetW();
		unsigned int iHeight 	= SExp.at(iDevID)->f2dEdges.GetH();

		assert(iWidth 	> 0);
		assert(iHeight 	> 0);

		thrust::device_vector<float> dvRes(iWidth, 0.f);

		for(int y = 0; y < static_cast<int>(iHeight); y++) {               
			thrust::transform(SExp.at(iDevID)->f2dEdges.GetRowBegin(y),
				SExp.at(iDevID)->f2dEdges.GetRowEnd(y),
				dvRes.begin(),
				dvRes.begin(),
				spowAmXpY_functor((*SExp.at(iDevID)->dvInput)[y]) );
		}

		if(fConscRate > 0.f) { 								// Implementation of conscience mechanism
			thrust::transform(dvRes.begin(),					// input
				dvRes.end(),							// input
				SExp.at(iDevID)->dvConscience->begin(),				// input
				dvRes.begin(),							// result
				sXmAmY_functor(1.f/(float)iWidth) );				// functor

			thrust::transform(dvRes.begin(),					// input
				dvRes.end(),							// input
				SExp.at(iDevID)->dvConscience->begin(),				// input
				SExp.at(iDevID)->dvConscience->begin(),				// result
				sAXmY_functor(fConscRate) );					// functor
		}

		std::pair<float, unsigned int> pCurBMUVal = devGetMin(dvRes);
		BMUExport BMU(pCurBMUVal.first, pCurBMUVal.second, iDevID);
		vBMUExp[iDevID] = BMU;
	}

	resBMU = hostGetMin(vBMUExp);
	checkCudaErrors(cudaSetDevice(resBMU.iDeviceID) );
	resBMU.dvBMUPos = SExp.at(resBMU.iDeviceID)->f2dPositions.GetSubArrayY(resBMU.iBMUID);

	return resBMU;
}

/*
 * Layout of SOMPositionF2DArray:
 * 		COL1	COL2	COL3	COL(n+1)
 * ROW1		Xpos	Xpos	Xpos	..
 * ROW2		Ypos	Ypos	Ypos	..
 * ROW3		Zpos	Zpos	Zpos	..
 * ROW(n+1)	..		..		..		..
 */
template<typename BinaryFunction>
void hostSOMPropagateBW( std::vector<SOMExport*> &SExp,
		const BMUExport &BMU,
		const unsigned int &fCycle,
		const unsigned int &fCycles,
		BinaryFunction binaryDistFunc
		)
{
	omp_set_num_threads(SExp.size() );  							// create as many CPU threads as there are CUDA devices
	#pragma omp parallel 									// for(int iDev = 0; iDev < static_cast<int>(SExp.size() ); iDev++) {
	{
		unsigned int iDevID 	= omp_get_thread_num();
		checkCudaErrors(cudaSetDevice(iDevID) );
		
		unsigned int iWidth 	= SExp.at(iDevID)->f2dPositions.GetW();
		unsigned int iHeight 	= SExp.at(iDevID)->f2dPositions.GetH();

		thrust::device_vector<float> dvTmp (iWidth, 0.f); 				// temporary
		thrust::device_vector<float> dvLearningRate(iWidth, 0.f);
		thrust::device_vector<float> dvInfl(iWidth, 0.f);
		thrust::device_vector<float> dvDist(iWidth, 0.f);
		
		// 1. Calc distances for all neurons to BMNeuron: Distance = sqrt(pow(x,2)+pow(y,2)+pow(z,2)+pow(n+1,2) )
		for(int y = 0; y < static_cast<int>(iHeight); y++) { 				// for each coordinate position of the neuron
			thrust::transform(
				SExp.at(iDevID)->f2dPositions.GetRowBegin(y),
				SExp.at(iDevID)->f2dPositions.GetRowEnd(y),
				dvDist.begin(),
				dvDist.begin(),
				spowAmXpY_functor(BMU.dvBMUPos[y]) );
		}

                thrust::transform(dvDist.begin(), dvDist.end(), dvDist.begin(), square_root());
		
		// 1 b calc learning rate
		thrust::device_vector<float> *dvLRate = SExp.at(iDevID)->dvLearningRate;
		thrust::transform( dvLRate->begin(),						// input
			dvLRate->end(), 							// input
			dvLearningRate.begin(), 						// result
			sm13lrate_decay_functor(fCycle, fCycles) );				// functor
			
		// 1 c Calc SigmaT
		thrust::device_vector<float> *dvSigma0 = SExp.at(iDevID)->dvSigma0;
		thrust::transform( dvSigma0->begin(),						// input
			dvSigma0->end(), 							// input
			dvTmp.begin(), 								// result
			sm13rad_decay_functor(fCycle, fCycles) );				// functor
		
		// 2. Calculate the influence for each neuron
		thrust::transform( dvTmp.begin(),						// input
			dvTmp.end(), 								// input
			dvDist.begin(), 							// input 2
			dvInfl.begin(), 							// result
			binaryDistFunc );							// functor

		// 2 b
		thrust::transform( dvInfl.begin(),						// input
			dvInfl.end(), 								// input
			dvLearningRate.begin(), 						// input 2
			dvInfl.begin(), 							// result
			thrust::multiplies<float>() );								// functor

		// 3. Only handle neurons in radius:
		// 3a. Make stencil
		thrust::transform( dvDist.begin(), 						// input
			dvDist.end(),								// input
			dvTmp.begin(),								// input 2
			dvTmp.begin(), 								// result
			thrust::less<float>() 							// functor
		);
		// 3b. Use stencil to modify only neurons inside the radius
		iWidth 	= SExp.at(iDevID)->f2dEdges.GetW();
		iHeight = SExp.at(iDevID)->f2dEdges.GetH();
		for(int y = 0; y < static_cast<int>(iHeight); y++) {				// for each edge of the neuron
			thrust::transform_if( SExp.at(iDevID)->f2dEdges.GetRowBegin(y),		// input 1
				SExp.at(iDevID)->f2dEdges.GetRowEnd(y), 			// input 1
				dvInfl.begin(),							// input 2
				dvTmp.begin(),							// stencil
				SExp.at(iDevID)->f2dEdges.GetRowBegin(y), 			// result
				hebbian_functor((*SExp.at(iDevID)->dvInput)[y]), // functor
				thrust::identity<int>() ); 					// predicate
		}
	}
}

void hostSOMTrainHelper( std::vector<SOMExport*> &SExp,
		const ANN::TrainingSet &InputSet,
		const unsigned int &iCycles, 
		const float &fConscRate,
		const ANN::DistFunction &DistFunc, 
		const unsigned int &iPatternID,
		const unsigned int &iCycle) 
{
	assert(iPatternID < InputSet.GetNrElements() );

	// Set Input
	std::vector<float> vCurInput = InputSet.GetInput(iPatternID);
	for(int iDevID = 0; iDevID < static_cast<int>(SExp.size() ); iDevID++) {
		checkCudaErrors(cudaSetDevice(iDevID) );

		thrust::device_vector<float> *p_dvInputVector = new thrust::device_vector<float>(vCurInput.size() );
		thrust::copy(vCurInput.begin(), vCurInput.end(), p_dvInputVector->begin() );
		SExp[iDevID]->dvInput = p_dvInputVector;
	}

	// Find BMNeuron 
	BMUExport BMUExp = hostSOMFindBMNeuronID(SExp, fConscRate);

	// Propagate BW SM 2.0
	hostSOMPropagateBW( SExp,
		BMUExp,									// const
		iCycle,
		iCycles,
		sm20distance_functor(DistFunc.distance)); 				// const
}

void hostSOMTraining( std::vector<SOMExport*> &SExp,
		const ANN::TrainingSet &InputSet,
		const unsigned int &iCycles, 
		const float &fConscRate,
		const ANN::DistFunction &DistFunc,
		const ANN::TrainingMode &eMode )
{
	int iMin 		= 0;
	int iMax 		= InputSet.GetNrElements()-1;
	int iProgCount 		= 1;

	for(int iCycle = 0; iCycle < static_cast<int>(iCycles); iCycle++) {
		if(iCycles >= 10) {
			if(((iCycle+1) / (iCycles/10)) == iProgCount && (iCycle+1) % (iCycles/10) == 0) {
				std::cout<<"Current training progress calculated by the GPU is: "<<iProgCount*10.f<<"%/Step="<<iCycle+1<<std::endl;
				iProgCount++;
			}
		} 
		else {
			std::cout<<"Current training progress calculated by the CPU is: "<<(float)(iCycle+1.f)/(float)iCycles*100.f<<"%/Step="<<iCycle+1<<std::endl;
		}

		if(eMode == ANN::ANRandomMode) {
			unsigned int iRandID = ANN::RandInt(iMin, iMax);
			hostSOMTrainHelper(SExp, InputSet, iCycles, fConscRate, DistFunc, iRandID, iCycle);
		}
		// The input vectors are presented to the network in serial order
		else if(eMode == ANN::ANSerialMode) {
			for(unsigned int j = 0; j < InputSet.GetNrElements(); j++) {
				hostSOMTrainHelper(SExp, InputSet, iCycles, fConscRate, DistFunc, j, iCycle);
			}
		}
	}
}

#endif
