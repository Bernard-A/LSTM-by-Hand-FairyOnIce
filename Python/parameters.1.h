//
// Generated by spiderweak using Python.
//

#ifndef CPP_PARAMETERS_H
#define CPP_PARAMETERS_H

#define HUNIT 1

#endif //CPP_PARAMETERS_H

const int hunit = HUNIT;

const float lstm_cell_input_weights[4 * HUNIT] = {-0.806347668170929, 0.06758153438568115, 0.1901506930589676, -0.5765173435211182};

const float lstm_cell_hidden_weights[4 * HUNIT * HUNIT] = {-0.5551999807357788, -0.8021552562713623, 0.2173292189836502, -0.032678913325071335};

const float lstm_cell_bias[4 * HUNIT] = {0.40350955724716187, 1.0, -0.477422297000885, 0.42344745993614197};

static float lstm_cell_hidden_layer[HUNIT] = {-0.4801156222820282};
static float lstm_cell_cell_states[HUNIT] = {-1.3535175323486328};

const float dense_weights[HUNIT] = {-0.9467102885246277};
const float dense_bias = 0.34821823239326477;
