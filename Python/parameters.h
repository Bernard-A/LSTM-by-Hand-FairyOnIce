//
// Generated by spiderweak using Python.
//

#ifndef CPP_PARAMETERS_H
#define CPP_PARAMETERS_H

#define HUNIT 1

#endif //CPP_PARAMETERS_H

const int hunit = HUNIT;

const float lstm_cell_input_weights[4 * HUNIT] = {0.7236478924751282, -0.7236429452896118, -0.011296585202217102, 0.9588750004768372};

const float lstm_cell_hidden_weights[4 * HUNIT * HUNIT] = {0.7760595083236694, -0.22524213790893555, 0.27519720792770386, -0.5208302736282349};

const float lstm_cell_bias[4 * HUNIT] = {0.2248043715953827, 1.0, -0.27169960737228394, 0.2257799506187439};

float lstm_cell_hidden_layer[HUNIT] = {-0.40004485845565796};
float lstm_cell_cell_states[HUNIT] = {-0.6508712768554688};

const float dense_weights[HUNIT] = {-1.345304012298584};
const float dense_bias = 0.2549245059490204;
