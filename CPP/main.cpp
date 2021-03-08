//
// Created by spiderweak on 1/21/21.
//

#include <iostream>
#include <cmath>
#define PI 3.141592654

struct LSTMCell {
    float y_t_1; // Previous output
    float c_t_1; // Previous Cell State
    float W_i_x; // Weight i against x
    float W_f_x; // Weight f against x
    float W_z_x; // Weight z against x
    float W_o_x; // Weight o against x
    float W_i_y; // Weight i against y
    float W_f_y; // Weight f against y
    float W_z_y; // Weight z against y
    float W_o_y; // Weight o against y
    float b_i; // Bias i
    float b_f; // Bias f
    float b_z; // Bias z
    float b_o; // Bias o
};

float dense_nn(float input, float W, float b);
float complete_nn(float input, float * weights_array);
float lstm_chain(float input_0, int layer_number, float * cells_data_array);
void lstm_cell (float x_t, struct LSTMCell lstmCell, float* output);
float sigmoid_function (float input);
float dot(float a, float b);
float g(float input);
float h(float input);

int main() {

    float weight_array[] = {-0.218822, -0.011803,
                            1.1231076, 0.4633199, -1.190576 , -0.0426245,
                            0.45407116, 0.00370111, 0.3802879 , 0.8057212,
                            0.18706907,  1.        , -0.22962211,  0.14464839};

// Yt-1 = 0.449882 => -0.2188218818202041 , Yt = 0.4286432 => -0.2020145486608096, xt = 0.428020

// [array([[ 1.1231076,  0.4633199, -1.190576 , -0.0426245]], dtype=float32)
//  array([[0.45407116, 0.00370111, 0.3802879 , 0.8057212 ]], dtype=float32),
//  array([ 0.18706907,  1.        , -0.22962211,  0.14464839], dtype=float32)]

// Program initialized with y_t_1 = -0.2188218818202041 and c_t_1 = -0.011803 for an x_t of 0.428020

    float value = complete_nn(0.428020, weight_array);

    printf("Final Value %f\n", value);

    return 0;
}

float complete_nn(float input, float * weights_array) {//  Will need complete declaration }, float **network) {

    float intermediary = input;
    float final = input;
    int number_of_lstm_layers = 1; // Hardcoded, will need actual values

    intermediary = lstm_chain(input, number_of_lstm_layers, weights_array);

    /** Need to change this to fit parameters **/
    float dense_weight = -1.2636627;
    float dense_bias = 0.17336495;

    final = dense_nn(intermediary, dense_weight, dense_bias);

    return final;

}

float dense_nn(float input, float W, float b) {
    return input * W + b;
}

float lstm_chain(float input_0, int layer_number, float * cells_data_array) {

    float input = input_0;

    struct LSTMCell lstmCellinChain{};

    float output[2] = {0.,0.};

    for (int i = 0; i < layer_number; ++i) {
        lstmCellinChain.y_t_1 = cells_data_array[i * 14 + 0];
        lstmCellinChain.c_t_1 = cells_data_array[i * 14 + 1];

        lstmCellinChain.W_i_x = cells_data_array[i * 14 + 2];
        lstmCellinChain.W_f_x = cells_data_array[i * 14 + 3];
        lstmCellinChain.W_z_x = cells_data_array[i * 14 + 4];
        lstmCellinChain.W_o_x = cells_data_array[i * 14 + 5];

        lstmCellinChain.W_i_y = cells_data_array[i * 14 + 6];
        lstmCellinChain.W_f_y = cells_data_array[i * 14 + 7];
        lstmCellinChain.W_z_y = cells_data_array[i * 14 + 8];
        lstmCellinChain.W_o_y = cells_data_array[i * 14 + 9];

        lstmCellinChain.b_i = cells_data_array[i * 14 + 10];
        lstmCellinChain.b_f = cells_data_array[i * 14 + 11];
        lstmCellinChain.b_z = cells_data_array[i * 14 + 12];
        lstmCellinChain.b_o = cells_data_array[i * 14 + 13];

        lstm_cell(input, lstmCellinChain, output);

        cells_data_array[i * 14+0] = output[0];
        cells_data_array[i * 14+1] = output[1];

        input = output[0];
        // No changes to input;
    }
    return output[0];
}

void lstm_cell (float x_t, struct LSTMCell lstmCell, float* output) {

    float c_t = 0.;
    float y_t = 0.;

    float z_t = 0.;
    float i_t = 0.;
    float f_t = 0.;
    float o_t = 0.;

    /**
     * TODO :
     * // - Implement g and h function
     * // - Pass Weight & bias as parameters Done
     */

    // Forget Gate
    f_t = sigmoid_function(lstmCell.W_f_x * x_t + lstmCell.W_f_y * lstmCell.y_t_1 + lstmCell.b_f);

    // Input Gate
    i_t = sigmoid_function(lstmCell.W_i_x * x_t + lstmCell.W_i_y * lstmCell.y_t_1 + lstmCell.b_i);

    // Current cell state
    z_t = g(lstmCell.W_z_x * x_t + lstmCell.W_z_y * lstmCell.y_t_1 + lstmCell.b_z);

    // Final Cell state
    c_t = dot(z_t, i_t) + dot(lstmCell.c_t_1, f_t);

    // Output Cell state
    o_t = sigmoid_function(lstmCell.W_o_x * x_t + lstmCell.W_o_y * lstmCell.y_t_1 + lstmCell.b_o);

    // Result
    y_t = dot(h(c_t), o_t);

    output[0] = y_t;
    output[1] = c_t;
}


float sigmoid_function (float input) {
    return 1/(1+((float) exp(- (double) input))); // 1/(1+exp(-(input)));
}

float dot( float a, float b) {
    return a*b;
}

float g(float input) { // Supposed to be tanh(input) from readings
    return (float) tanh((double) input);
}

float h(float input) {
    return (float) tanh((double) input); // Supposed to be tanh(input) from readings
}