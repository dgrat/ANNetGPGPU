/*
 * main.cpp
 *
 *  Created on: 12.04.2010
 *      Author: dgrat
 */

#include <ANNet>
#include <ANContainers>
#include <ANMath>

#include "Samples.h"



int main(int argc, char *argv[]) {
	ANN::TrainingSet<float> input;
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
	int w1 = 32;
	int w2 = 4;

	ANN::SOMNet<float, ANN::functor_gaussian<float>> cpu;
	cpu.CreateSOM(3, 1, w1,w1);
	cpu.SetTrainingSet(input);
	cpu.Training(1, ANN::ANSerialMode);

	return 0;
}
