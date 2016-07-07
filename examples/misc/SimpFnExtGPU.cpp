/*
 * main.cpp
 *
 *  Created on: 12.04.2010
 *      Author: dgrat
 */

#include <ANNet>
#include <ANGPGPU>
#include <ANContainers>
#include <ANMath>

#include "QSOMReader.h"
#include "Samples.h"

#include "SetFcn.h"

#include <ctime>
#include <iostream>

using namespace ANN;
using namespace ANNGPGPU;

ANN::DistFunction ownFn = {
	(char*)"own",
	NULL,
	fcn_rad_decay,
	fcn_lrate_decay
};

int main(int argc, char *argv[]) {
	QApplication a(argc, argv);

	TrainingSet input;
	input.AddInput(red);
	input.AddInput(green);
	input.AddInput(dk_green);
	input.AddInput(blue);
	input.AddInput(dk_blue);
	input.AddInput(yellow);
	input.AddInput(orange);
	input.AddInput(purple);
	input.AddInput(black);
	input.AddInput(white);

	std::vector<float> vCol(3);
	int w1 = 120;
	int w2 = 4;

	SOMNetGPU gpu;
	gpu.CreateSOM(3, 1, w1,w1);
	gpu.SetTrainingSet(input);
	
	gpu.SetLearningRate(0.1);
	//gpu.SetSigma0(25);
	gpu.Training(500);
        
	SetFcn(&ownFn);
	gpu.SetDistFunction(ownFn);
	// or just: SetFcn(gpu.GetDistFunction() );
        
        // Clear initial weights
        for(int x = 0; x < w1*w1; x++) {
            ANN::SOMNeuron *pNeur = (ANN::SOMNeuron*)((ANN::SOMLayer*)gpu.GetOPLayer())->GetNeuron(x);
            pNeur->GetConI(0)->SetValue(0); 
            pNeur->GetConI(1)->SetValue(0); 
            pNeur->GetConI(2)->SetValue(0); 
            // Except for one unit.
            if (x == 820) {
                pNeur->GetConI(0)->SetValue(1); 
                pNeur->GetConI(1)->SetValue(1); 
                pNeur->GetConI(2)->SetValue(1); 
            }
        }
        
	gpu.Training(1);

	SOMReader w(w1, w1, w2);
	for(int x = 0; x < w1*w1; x++) {
		SOMNeuron *pNeur = (SOMNeuron*)((SOMLayer*)gpu.GetOPLayer())->GetNeuron(x);
		vCol[0] = pNeur->GetConI(0)->GetValue();
		vCol[1] = pNeur->GetConI(1)->GetValue();
		vCol[2] = pNeur->GetConI(2)->GetValue();

		w.SetField(QPoint(pNeur->GetPosition()[0], pNeur->GetPosition()[1]), vCol );
	}
	w.Save("SimpFnExtByGPU.png");
	return 0;
}
