//
// Generated by spiderweak using Python.
//

#ifndef CPP_PARAMETERS_H
#define CPP_PARAMETERS_H

#define HUNIT 1

#endif //CPP_PARAMETERS_H

const int hunit = HUNIT;

const float lstm_cell_input_weights[4 * HUNIT] = {0.11850305646657944, -0.27645057439804077, 0.017958013340830803, -1.2644069194793701};

const float lstm_cell_hidden_weights[4 * HUNIT * HUNIT] = {-0.6233807802200317, 0.13200156390666962, -0.7242480516433716, -0.2635084390640259};

const float lstm_cell_bias[4 * HUNIT] = {0.8864936828613281, 1.0, -0.870543897151947, 0.5227345824241638};

float lstm_cell_hidden_layer[HUNIT] = {-0.4616917371749878};
float lstm_cell_cell_states[HUNIT] = {-1.2524135112762451};

const float dense_weights[HUNIT] = {-0.6404330730438232};
const float dense_bias = 0.3013148605823517;
