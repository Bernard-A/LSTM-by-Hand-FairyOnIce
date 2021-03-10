/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <stdio.h>

#include "mbed.h"
#include "rtos.h"

#include "lorawan/LoRaWANInterface.h"
#include "lorawan/system/lorawan_data_structures.h"
#include "events/EventQueue.h"

// Application helpers
#include "lora_radio_helper.h"

// For Math applications
#include <math.h>
#define PI 3.141592654

// Personal functions LSTM By Hand
#include "handmade.h"

// Data for prediction
#include "conso_data.h"

// LSTM Parameters
#include "parameters.h"

using namespace events;

// Max payload size can be LORAMAC_PHY_MAXPAYLOAD.
// This example only communicates with much shorter messages (<30 bytes).
// If longer messages are used, these buffers must be changed accordingly.
uint8_t tx_buffer[30];
uint8_t rx_buffer[30];

/*
 * Setting up transmission threshold
 */
#define THRESHOLD			0.3

#define TX_INTERVAL             3000
#define MINIMUM_CONFIDENCE      0.7

/*
 * Sets up an application dependent transmission timer in ms. Used only when Duty Cycling is off for testing
 */
#define TX_TIMER                        10000

/**
 * Maximum number of events for the event queue.
 * 10 is the safe number for the stack events, however, if application
 * also uses the queue for whatever purposes, this number should be increased.
 */
#define MAX_NUMBER_OF_EVENTS            10

/**
 * Maximum number of retries for CONFIRMED messages before giving up
 */
#define CONFIRMED_MSG_RETRY_COUNTER     3

#define MBED_CONF_LORA_DUTY_CYCLE_ON true

/**
* This event queue is the global event queue for both the
* application and stack. To conserve memory, the stack is designed to run
* in the same thread as the application and the application is responsible for
* providing an event queue to the stack that will be used for ISR deferment as
* well as application information event queuing.
*/
static EventQueue ev_queue(MAX_NUMBER_OF_EVENTS *EVENTS_EVENT_SIZE);

/**
 * Event handler.
 *
 * This will be passed to the LoRaWAN stack to queue events for the
 * application which in turn drive the application.
 */
static void lora_event_handler(lorawan_event_t event);

/**
 * Constructing Mbed LoRaWANInterface and passing it the radio object from lora_radio_helper.
 */
static LoRaWANInterface lorawan(radio);

/**
 * Application specific callbacks
 */
static lorawan_app_callbacks_t callbacks;

int inference_count = 0;

/**
 * Entry point for application
 */
int main(void) {

    // Keep track of how many inferences we have performed.
    inference_count = 0;

    // stores the status of a call to LoRaWAN protocol
    lorawan_status_t retcode;

    // Initialize LoRaWAN stack
    if (lorawan.initialize(&ev_queue) != LORAWAN_STATUS_OK) {
        printf("\r\n LoRa initialization failed! \r\n");
        return -1;
    }

    printf("\r\n Mbed LoRaWANStack initialized \r\n");

    // prepare application callbacks
    callbacks.events = mbed::callback(lora_event_handler);
    lorawan.add_app_callbacks(&callbacks);

    // Set number of retries in case of CONFIRMED messages
    if (lorawan.set_confirmed_msg_retries(CONFIRMED_MSG_RETRY_COUNTER)
            != LORAWAN_STATUS_OK) {
        printf("\r\n set_confirmed_msg_retries failed! \r\n\r\n");
        return -1;
    }

    printf("\r\n CONFIRMED message retries : %d \r\n",
           CONFIRMED_MSG_RETRY_COUNTER);

    // Enable adaptive data rate
    if (lorawan.enable_adaptive_datarate() != LORAWAN_STATUS_OK) {
        printf("\r\n enable_adaptive_datarate failed! \r\n");
        return -1;
    }

    printf("\r\n Adaptive data  rate (ADR) - Enabled \r\n");

    retcode = lorawan.connect();

    if (retcode == LORAWAN_STATUS_OK ||
            retcode == LORAWAN_STATUS_CONNECT_IN_PROGRESS) {
    } else {
        printf("\r\n Connection error, code = %d \r\n", retcode);
        return -1;
    }

    printf("\r\n Connection - In Progress ...\r\n");

    // make your event queue dispatching events forever
    ev_queue.dispatch_forever();

    return 0;      
}

/**
 * Sends a message to the Network Server
 */
static void send_message()
{
    printf("waiting 7 secs");
    ThisThread::sleep_for(chrono::seconds(7));
    printf("waited 7s");

    // 0. Buffer allocation
    static uint8_t last_top_result = 99;
    static bool first_send_message = true;
    bool predict_nok;
    static float previously_transmitted;
    static int index_value;
    static int skipped;

    float *lstm_output;
    float output_value;

    // Loading data for tests if first time booting
    if (first_send_message) {
            previously_transmitted = conso_data[0];//load_data_Init();
            first_send_message = false;
            predict_nok = true;
            index_value = 0;
            skipped = 0;

    // 1. Import data into buffer
    float x_min = -363.16381836;
    float x_max = 373.3527832;
    float tx_min = 0.;
    float tx_max = 0.9;
    // Change this value to fit value for inferencing
    float x_val = (previously_transmitted - x_min) / (x_max - x_min) * (tx_max - tx_min) + tx_min;
            // X_min= -363.16381836
            // X_max= +373.3527832
            // max=0.9
            // min=0
            // Formula : X_scaled = (X-X_min)/(X_max-X_min) * (max-min) + min

    // Dual prediction
    // We position ourselves as the server and base prediction on previously transmitted value


    lstm_output = lstmCellSimple(x_val, lstm_cell_input_weights, lstm_cell_hidden_weights,
                                 lstm_cell_bias, lstm_cell_hidden_layer, lstm_cell_cell_states);

    output_value = dense_nn(lstm_output, dense_weights, dense_bias);

    float y_val = output_value;

    // Determining result unscaled
    y_val = (y_val - tx_min) / (tx_max - tx_min) * (x_max - x_min) + x_min + previously_transmitted;

    printf("Index Value %i\n", index_value);

    printf("Value calculated is %i\n",(int) (y_val));

    index_value++;

    if (index_value == 720) { index_value = 0;}
    
    printf("Actual data was : %i\n", (int)(conso_data[index_value]));
   
    float difference_prediction = (y_val-conso_data[index_value])/(conso_data[index_value]);

    if (difference_prediction < 0) {
            difference_prediction = - difference_prediction;
    }

    if (difference_prediction < THRESHOLD) {
        predict_nok=false;
        skipped++;
    } else {
        predict_nok=true;
        y_val=conso_data[index_value];
    } 
    
    // 3. Log the prediction

    printf("Data to transmit : %i \n",(int)(y_val));

    // 4. Send data based on prediction

    previously_transmitted = y_val;

    uint16_t packet_len;
    int16_t retcode;

    if (predict_nok) {
        packet_len = sprintf((char *) tx_buffer, "Transmitted Value is %d skip %d",
            (int) (y_val), skipped);
        retcode = lorawan.send(MBED_CONF_LORA_APP_PORT, tx_buffer, packet_len,
                           MSG_UNCONFIRMED_FLAG);

        if (retcode < 0) {
            retcode == LORAWAN_STATUS_WOULD_BLOCK ? printf("send - Duty cycle violation\r\n")
                : printf("send - Error code %d \r\n", retcode);

            if (retcode == LORAWAN_STATUS_WOULD_BLOCK) {
                // retry in 3 secs
                ev_queue.call_in(10000, send_message);
            }
            return;
        }
        printf("%d bytes scheduled for transmission \r\n", retcode);
        memset(tx_buffer, 0, sizeof(tx_buffer));
        skipped = 0;
    } else {
        send_message();
    }
}

/**
 * Receive a message from the Network Server
 */
static void receive_message()
{
    uint8_t port;
    int flags;
    int16_t retcode = lorawan.receive(rx_buffer, sizeof(rx_buffer), port, flags);

    if (retcode < 0) {
        printf("\r\n receive() - Error code %d \r\n", retcode);
        return;
    }

    printf(" RX Data on port %u (%d bytes): ", port, retcode);
    for (uint8_t i = 0; i < retcode; i++) {
        printf("%02x ", rx_buffer[i]);
    }
    printf("\r\n");

    memset(rx_buffer, 0, sizeof(rx_buffer));
}

/**
 * Event handler
 */
static void lora_event_handler(lorawan_event_t event)
{
    switch (event) {
        case CONNECTED:
            printf("\r\n Connection - Successful \r\n");
            if (MBED_CONF_LORA_DUTY_CYCLE_ON) {
                send_message();
            } else {
                ev_queue.call_every(TX_TIMER, send_message);
            }

            break;
        case DISCONNECTED:
            ev_queue.break_dispatch();
            printf("\r\n Disconnected Successfully \r\n");
            break;
        case TX_DONE:
            printf("\r\n Message Sent to Network Server \r\n");
            if (MBED_CONF_LORA_DUTY_CYCLE_ON) {
                send_message();
            }
            break;
        case TX_TIMEOUT:
        case TX_ERROR:
        case TX_CRYPTO_ERROR:
        case TX_SCHEDULING_ERROR:
            printf("\r\n Transmission Error - EventCode = %d \r\n", event);
            // try again
            if (MBED_CONF_LORA_DUTY_CYCLE_ON) {
                send_message();
            }
            break;
        case RX_DONE:
            printf("\r\n Received message from Network Server \r\n");
            receive_message();
            break;
        case RX_TIMEOUT:
        case RX_ERROR:
            printf("\r\n Error in reception - Code = %d \r\n", event);
            break;
        case JOIN_FAILURE:
            printf("\r\n OTAA Failed - Check Keys \r\n");
            break;
        case UPLINK_REQUIRED:
            printf("\r\n Uplink required by NS \r\n");
            if (MBED_CONF_LORA_DUTY_CYCLE_ON) {
                send_message();
            }
            break;
        default:
            MBED_ASSERT("Unknown Event");
    }
}

float * lstmCellSimple(float input, const float * input_weights, const float * hidden_weights,
                       const float * bias, float * hidden_layer, const float * cell_states) {
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

    hidden_layer = new_hidden_layer;
    cell_states = new_cell_states;

    return hidden_layer;
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
    return 1/(1+(exp(-input)));
}
//EOF
