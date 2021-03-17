//
// Created by spiderweak on 3/10/21.
//

/**
 * How to use this file :
 *
 * Set HUNIT to the numbers of hidden layers for your LSTM
 *
 * Then set the weights as extracted from Python's get_weights function on your LSTM
 *
 * If you wish to redevelop on your side, note the importance of this order in the weights
 * 	InputCell, ForgetCell, CellCandidate, OutputCell
 *
 * The get_weights function returns a given amount of array
 * 	First array is input_weights of size 4*HUNIT
 * 	Second array is hidden_weights, a matrix of size HUNIT x 4*HUNIT
 * 	Third array is bias of size 4*HUNIT
 */

#ifndef CPP_PARAMETERS_H
#define CPP_PARAMETERS_H

#define HUNIT 3

#endif //CPP_PARAMETERS_H

const int hunit = HUNIT;

const float lstm_cell_input_weights[4 * HUNIT] = {0.3234084, 0.70131856, 0.37922043,
                                                  0.66995215, -0.54794693, 0.1504041,
                                                  -0.10613328, -0.1059448, -0.6422383,
                                                  -0.00922386, 0.08984761, 0.14972672};

const float lstm_cell_hidden_weights[4 * HUNIT * HUNIT] = {0.4997586, -0.42921698, 0.24008259,
                                                           0.03498572, -0.13131998, 0.15442386,
                                                           -0.33034173, -0.0227247, -0.35828012,
                                                           -0.1899443, 0.21435145, -0.38213438,
                                                           0.06887937, 0.24827306, 0.00793623,
                                                           0.5819455, 0.269377, 0.24519907,
                                                           0.2678546, -0.4108627, 0.19137171,
                                                           -0.28056607, 0.19283584, -0.2629174,
                                                           -0.1054808, 0.03056782, 0.10177574,
                                                           0.22523263, -0.27819708, -0.60721684,
                                                           -0.11857513, -0.12831974, -0.09951104,
                                                           -0.43784, 0.33351612, 0.37064832};

const float lstm_cell_bias[4 * HUNIT] = {0.10343148, 0.11245319, 0.21322289,
                                         1., 1., 1.,
                                         0.18435329, 0.18004698, -0.2149152,
                                         0.09632485, 0.0977062, 0.20585288};

float lstm_cell_hidden_layer[HUNIT] = {-0.12745334, 0.06317158, -0.15865295};
float lstm_cell_cell_states[HUNIT] = {-0.2454686, 0.12659174, -0.32833787};

const float dense_weights[HUNIT] = {-0.87860644, 0.39441893, -1.188957};
const float dense_bias = 0.2822338;


