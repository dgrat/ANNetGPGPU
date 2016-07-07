/*
 * main.cpp
 *
 *  Created on: 12.04.2010
 *      Author: dgrat
 */

#include <ANNet>
#include <ANContainers>
#include <ANMath>

#include "QSOMReader.h"
#include "Samples.h"

#include <ctime>
#include <iostream>


int main(int argc, char *argv[]) {
	QApplication a(argc, argv);

	ANN::TrainingSet input;
	input.AddInput(red);

	std::vector<float> vCol(3);
	int w1 = 64;
	int w2 = 4;

	ANN::SOMNet cpu;
        cpu.SetLearningRate(1);
	cpu.CreateSOM(3, 1, w1,w1);
	cpu.SetTrainingSet(input);

        // Clear initial weights
        for(int x = 0; x < w1*w1; x++) {
            ANN::SOMNeuron *pNeur = (ANN::SOMNeuron*)((ANN::SOMLayer*)cpu.GetOPLayer())->GetNeuron(x);
            pNeur->GetConI(0)->SetValue(0); 
            pNeur->GetConI(1)->SetValue(0); 
            pNeur->GetConI(2)->SetValue(0); 
            // Except for one unit.
            if (x == 0) {
                pNeur->GetConI(0)->SetValue(0.5); 
                pNeur->GetConI(1)->SetValue(0.5); 
                pNeur->GetConI(2)->SetValue(0.5); 
            }
        }
        
        cpu.Training(1);

	SOMReader w(w1, w1, w2);
	for(int x = 0; x < w1*w1; x++) {
		ANN::SOMNeuron *pNeur = (ANN::SOMNeuron*)((ANN::SOMLayer*)cpu.GetOPLayer())->GetNeuron(x);
		vCol[0] = pNeur->GetConI(0)->GetValue();
		vCol[1] = pNeur->GetConI(1)->GetValue();
		vCol[2] = pNeur->GetConI(2)->GetValue();

		w.SetField(QPoint(pNeur->GetPosition()[0], pNeur->GetPosition()[1]), vCol );
	}
	w.Save("ColorsByCPU.png");
	return 0;
}
