//
// Created by spiderweak on 1/21/21.
//

#include <iostream>
#include <cmath>
#define PI 3.141592654
#include "parameters.h"

void lstmCellSimple(float input, const float * input_weights, const float * hidden_weights,
                       const float * bias, float * hidden_layer, float * cell_states);

float dense_nn(const float * input, const float * Weight, float bias);

float sigmoid_function (float input);

int main() {

    float input_value = 0.428020;
    // Yt-1 = 0.449882 => -0.2188218818202041 , Yt = 0.4286432 => -0.2020145486608096, xt = 0.428020
    float output_value;

    printf("%f\n", lstm_cell_hidden_layer[0]);

    lstmCellSimple(input_value, lstm_cell_input_weights, lstm_cell_hidden_weights,
                                 lstm_cell_bias, lstm_cell_hidden_layer, lstm_cell_cell_states);

    printf("%f\n", lstm_cell_hidden_layer[0]);

    output_value = dense_nn(lstm_cell_hidden_layer, dense_weights, dense_bias);

    printf("Output Value %f\n", output_value);

    return 0;
}

void lstmCellSimple(float input, const float * input_weights, const float * hidden_weights,
                       const float * bias, float * hidden_layer, float * cell_states) {
    /**
     * input - float
     * input_weight - float array (4*HUNIT) - Weights W_i, W_f, W_c, W_o
     * hidden_weights - float array (4*HUNIT*HUNIT) - Weights U_i, U_f, U_c, U_o
     * bias - float array (4*HUNIT) - Bias B_i, B_f, B_c, B_o
     * hidden_layer - float array (4*HUNIT) - Outputs h
     * cell_states - float array (4*HUNIT) - Cell states
     * HUNIT - size of hidden layer
     */

    float new_hidden_layer[HUNIT];
    float new_cell_states[HUNIT];

    float input_gate[HUNIT];
    float forget_gate[HUNIT];
    float cell_candidate[HUNIT];
    float output_gate[HUNIT];

    for (int i = 0; i < HUNIT; ++i) {
        input_gate[i] = input_weights[0 * HUNIT + i] * input;
        forget_gate[i] = input_weights[1 * HUNIT + i] * input;
        cell_candidate[i] = input_weights[2 * HUNIT + i] * input;
        output_gate[i] = input_weights[3 * HUNIT + i] * input;

        for (int j = 0; j < HUNIT; ++j) {
            input_gate[i] += hidden_weights[(0 * HUNIT + i) * HUNIT + j] * hidden_layer[j];
            forget_gate[i] += hidden_weights[(1 * HUNIT + i) * HUNIT + j] * hidden_layer[j];
            cell_candidate[i] += hidden_weights[(2 * HUNIT + i) * HUNIT + j] * hidden_layer[j];
            output_gate[i] += hidden_weights[(3 * HUNIT + i) * HUNIT + j] * hidden_layer[j];
        }

        input_gate[i] += bias[0 * HUNIT + i];
        forget_gate[i] += bias[1 * HUNIT + i];
        cell_candidate[i] += bias[2 * HUNIT + i];
        output_gate[i] += bias[3 * HUNIT + i];

        input_gate[i] = sigmoid_function(input_gate[i]);
        forget_gate[i] = sigmoid_function(forget_gate[i]);
        cell_candidate[i] = sigmoid_function(cell_candidate[i]);
        output_gate[i] = sigmoid_function(output_gate[i]);
    }

    for (int i = 0; i < HUNIT; ++i) {

        new_cell_states[i] = forget_gate[i] * cell_states [i] + input_gate[i] * cell_candidate[i];
        new_hidden_layer[i] = output_gate[i] * (float) (tanh((double) new_cell_states[i]));

    }

    for (int i = 0; i < HUNIT; ++i) {

        hidden_layer[i] = new_hidden_layer[i];
        cell_states[i] = new_cell_states[i];
    }

    return;
}

float dense_nn(const float * input, const float * Weight, float bias) {
    float output = 0;
    for (int i = 0; i < HUNIT; ++i) {
        output += input[i] * Weight[i];
    }
    output += bias;
    return output;
}


float sigmoid_function (float input) {
    return 1/(1+((float) exp(- (double) input))); // 1/(1+exp(-(input)));
}