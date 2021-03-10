
float dense_nn(float input, float W, float b);
float complete_nn(float input, float * weights_array);
float lstm_chain(float input_0, int layer_number, float * cells_data_array);
void lstm_cell (float x_t, struct LSTMCell lstmCell, float* output);
void retro_fitting(float y_t_1, float c_t_1, float * weight_array, float x_val, float y_val);
float sigmoid_function (float input);
float dot(float a, float b);
float g(float input);
float h(float input);
float tanh_minus_1 (float input);
