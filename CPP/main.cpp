//
// Created by spiderweak on 1/21/21.
//

#include <iostream>
#include <cmath>
#define PI 3.141592654
#include "parameters.h"

float * lstmCellSimple(float input, const float * input_weights, const float * hidden_weights,
                       const float * bias, float * hidden_layer, const float * cell_states, int hunit);

float dense_nn(const float * input, const float * Weight, float bias, int hunit);

float sigmoid_function (float input);

int main() {

    float * lstm_output;
    float input_value = 0.410709;
    float output_value;

    lstm_output = lstmCellSimple(input_value, lstm_cell_input_weights, lstm_cell_hidden_weights,
                                 lstm_cell_bias, lstm_cell_hidden_layer, lstm_cell_cell_states, hidden_unit);

    output_value = dense_nn(lstm_output, dense_weights, dense_bias, hidden_unit);

    printf("Output Value %f\n", output_value);

    return 0;
}

float * lstmCellSimple(float input, const float * input_weights, const float * hidden_weights,
                       const float * bias, float * hidden_layer, const float * cell_states, int hunit) {
    /**
     * input - float
     * input_weight - float array (4*hunit) - Weights W_i, W_f, W_c, W_o
     * hidden_weights - float array (4*hunit*hunit) - Weights U_i, U_f, U_c, U_o
     * bias - float array (4*hunit) - Bias B_i, B_f, B_c, B_o
     * hidden_layer - float array (4*hunit) - Outputs h
     * cell_states - float array (4*hunit) - Cell states
     * hunit - size of hidden layer
     */
    float new_hidden_layer[hunit];
    float new_cell_states[hunit];

    float input_gate[hunit];
    float forget_gate[hunit];
    float cell_candidate[hunit];
    float output_gate[hunit];

    for (int i = 0; i < hunit; ++i) {
        input_gate[i] = input_weights[0 * hunit + i] * input;
        forget_gate[i] = input_weights[1 * hunit + i] * input;
        cell_candidate[i] = input_weights[2 * hunit + i] * input;
        output_gate[i] = input_weights[3 * hunit + i] * input;

        for (int j = 0; j < hunit; ++j) {
            input_gate[i] += hidden_weights[(0 * hunit + i) * hunit + j] * hidden_layer[j];
            forget_gate[i] += hidden_weights[(1 * hunit + i) * hunit + j] * hidden_layer[j];
            cell_candidate[i] += hidden_weights[(2 * hunit + i) * hunit + j] * hidden_layer[j];
            output_gate[i] += hidden_weights[(3 * hunit + i) * hunit + j] * hidden_layer[j];
        }

        input_gate[i] += bias[0 * hunit + i];
        forget_gate[i] += bias[1 * hunit + i];
        cell_candidate[i] += bias[2 * hunit + i];
        output_gate[i] += bias[3 * hunit + i];

        input_gate[i] = sigmoid_function(input_gate[i]);
        forget_gate[i] = sigmoid_function(forget_gate[i]);
        cell_candidate[i] = sigmoid_function(cell_candidate[i]);
        output_gate[i] = sigmoid_function(output_gate[i]);
    }

    for (int i = 0; i < hunit; ++i) {

        new_cell_states[i] = forget_gate[i] * cell_states [i] + input_gate[i] * cell_candidate[i];
        new_hidden_layer[i] = output_gate[i] * (float) (tanh((double) new_cell_states[i]));

    }

    hidden_layer = new_hidden_layer;
    cell_states = new_cell_states;

    return hidden_layer;
}

float dense_nn(const float * input, const float * Weight, float bias, int hunit) {
    float output = 0;
    for (int i = 0; i < hunit; ++i) {
        output += input[i] * Weight[i];
    }
    output += bias;
    return output;
}


float sigmoid_function (float input) {
    return 1/(1+((float) exp(- (double) input))); // 1/(1+exp(-(input)));
}