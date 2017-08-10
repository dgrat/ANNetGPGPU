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
#include "AbsNet.h"

#include "BPNeuron.h"
#include "BPLayer.h"

#include "Common.h"
#include "containers/ConTable.h"
#include "containers/TrainingSet.h"
#include "containers/ConTable.h"

#include "Edge.h"

#include "math/Functions.h"
#include "math/Random.h"

#include <vector>
#include <string>

#include <iostream>
#include <cassert>
#include <algorithm>
#include <omp.h>
#endif

namespace ANN {

class Function;
template <class T> class ConTable;
template <class T, class F> class BPNeuron;
template <class T, class F> class BPLayer;

/**
 * \brief Implementation of a back propagation network.
 *
 * @author Daniel "dgrat" Frenzel
 */
template <class Type, class Functor>
class BPNet : public AbsNet<Type>
{
protected:	
	HebbianConf<Type> m_Setup;

	/**
	 * Adds a new layer to the network. New layer will get appended to m_lLayers.
	 * @param pLayer Pointer to the new layer.
	 */
	virtual void AddLayer(BPLayer<Type, Functor> *pLayer);

	
public:
	/**
	 * Standard constructor
	 */
	BPNet();
	/**
	 * Copy constructor for copying the complete network:
	 * @param pNet
	 */
	BPNet(BPNet<Type, Functor> *pNet);

	/*
	 *
	 */
	virtual void CreateNet(const ConTable<Type> &Net);

	/**
	 * Adds a layer to the network.
	 * @param iSize Number of neurons of the layer.
	 * @param flType Flag describing the type of the net.
	 */
	virtual BPLayer<Type, Functor> *AddLayer(const unsigned int &iSize, const LayerTypeFlag &flType);
	
	/**
	 * Cycles the input from m_pTrainingData
	 * Checks total error of the output returned from SetExpectedOutputData()
	 * @return Returns the total error of the net after every training step.
	 * @param iCycles Maximum number of training cycles
	 * @param fTolerance Maximum error value (working as a break condition for early break-off)
	 */
	virtual std::vector<Type> TrainFromData(const unsigned int &iCycles, const Type &fTolerance, const bool &bBreak, Type &fProgress);

	/**
	 * Will create a sub-network from layer "iStartID" to layer "iStopID".
	 * This network will have all the properties of the network it is derivated from,
	 * however without the layer of edges between "iStartID" and "iStopID".
	 * Also the first and last layers of the new sub-net will automatically get a new flag as input or output layer.
	 *
	 * @param iStartID
	 * @param iStopID
	 * @return Returns a pointer to the new sub-network.
	 */
	BPNet<Type, Functor> *GetSubNet(const unsigned int &iStartID, const unsigned int &iStopID);

	/**
	 * Define the learning rate, the weight decay and the momentum term.
	 */
	void Setup(const HebbianConf<Type> &config);
	
#ifdef __BPNet_ADDON
	#include __BPNet_ADDON
#endif
};

#include "BPNet.tpp"

}
