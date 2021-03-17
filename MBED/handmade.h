void lstmCellSimple(float input, const float * input_weights, const float * hidden_weights,
                       const float * bias, float * hidden_layer, float * cell_states);

float dense_nn(const float * input, const float * Weight, float bias);

float sigmoid_function (float input);
