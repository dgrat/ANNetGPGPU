/// -*- tab-width: 8; Mode: C++; c-basic-offset: 8; indent-tabs-mode: t -*-
/*
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
   
   Author: Daniel Frenzel (dgdanielf@gmail.com)
*/

#pragma once

#ifndef SWIG
#include <vector>
#include <map>
#include "containers/2DArrayGPU.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#endif


namespace ANNGPGPU {

template <class Type>
struct SOMExport {
// VARIABLES
	ANNGPGPU::F2DArray<Type> _f2dEdges;
	ANNGPGPU::F2DArray<Type> _f2dPositions;
        thrust::device_vector<Type> _dvSigma0;
	thrust::device_vector<Type> _dvLearningRate;
	
//FUNCTIONS
	SOMExport(const ANNGPGPU::F2DArray<Type> &mEdgeMat, 
		  const ANNGPGPU::F2DArray<Type> &mPosMat,
		  const thrust::host_vector<Type> &vSigma0,
		  const thrust::host_vector<Type> &vLearningRate);
        
	void SetSigma0(thrust::device_vector<Type> &dvSigma0);
	void Clear();
	
#ifdef __SOMExport_ADDON
	#include __SOMExport_ADDON
#endif
};

#include "SOMExport.tpp"

}
