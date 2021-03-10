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

// Personal functions LSTM By Hand
#include "handmade.h"

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


/**
 * For Math applications
 */
#include <math.h>
#define PI 3.141592654

/**
 * Simple LSTM Cell Structure
 */
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

const float conso_data[] = {173.0813, 210.26813, 191.95605 , 164.897   , 182.8583  , 155.27605 , 165.86206 ,
       145.82147 , 163.62976 , 157.1036  , 183.63597 , 150.56335 ,
       164.10635 , 134.9884  , 141.60818 , 156.79782 , 128.0335  ,
       139.9927  , 159.88116 , 164.71202 , 162.61412 , 144.54448 ,
       132.65446 , 133.62535 , 125.02608 , 117.649826, 130.96637 ,
       128.98077 , 115.817986, 125.533104, 121.99191 , 113.118286,
       129.4864  , 116.593254, 104.210365, 117.9344  , 125.85579 ,
       120.260826, 125.24759 , 127.10816 , 110.70083 , 127.304375,
       145.7845  , 118.403046, 158.15271 , 154.10442 , 147.138   ,
       179.78323 , 214.88875 , 231.35289 , 231.36711 , 246.52998 ,
       247.18672 , 249.18405 , 287.4233  , 287.81732 , 274.1872  ,
       246.8557  , 274.84616 , 266.50366 , 268.068   , 270.0303  ,
       292.871   , 287.7819  , 260.86005 , 291.5957  , 213.83049 ,
       286.8689  , 241.48103 , 227.94814 , 273.33765 , 259.46674 ,
       259.78473 , 243.82672 , 293.36374 , 308.372   , 265.16943 ,
       259.64545 , 278.889   , 335.3024  , 356.06073 , 291.54773 ,
       280.44296 , 300.21884 , 290.78207 , 300.9672  , 355.03967 ,
       336.5316  , 327.88287 , 306.39774 , 341.59064 , 301.75912 ,
       295.5832  , 335.0302  , 294.57355 , 310.02557 , 309.6007  ,
       301.02396 , 294.42184 , 316.6763  , 271.29837 , 292.1709  ,
       294.57196 , 309.95532 , 295.26672 , 336.4353  , 463.33395 ,
       311.058   , 418.25967 , 308.6943  , 419.77695 , 459.6374  ,
       335.7529  , 459.34692 , 290.6034  , 276.3835  , 328.43155 ,
       400.29233 , 471.3664  , 342.55325 , 341.95544 , 453.42343 ,
       347.6856  , 252.8992  , 257.84906 , 305.99615 , 279.34305 ,
       415.97717 , 243.47368 , 252.4022  , 264.71277 , 265.89346 ,
       275.0022  , 316.85782 , 238.98866 , 330.81525 , 263.38602 ,
       240.53902 , 231.56598 , 201.69652 , 198.143   , 227.16614 ,
       202.35457 , 218.64052 , 229.26088 , 228.77066 , 232.09059 ,
       181.12407 , 233.09567 , 222.55385 , 232.3756  , 204.2105  ,
       194.2527  , 179.31255 , 167.34569 , 192.0261  , 147.73344 ,
       125.49848 , 163.48776 , 161.09639 , 156.54698 , 183.44695 ,
       130.11394 , 125.49488 , 141.92322 , 132.92691 , 118.49874 ,
       153.35504 , 127.03988 , 123.00152 , 123.90882 , 127.93603 ,
       113.79648 , 120.21429 , 127.198494, 119.74268 , 122.18068 ,
       122.26947 , 129.19417 , 127.81348 , 138.55804 , 128.90245 ,
       135.23727 , 141.15584 , 138.942   , 150.55928 , 177.58403 ,
       211.59624 , 233.16362 , 256.86398 , 255.96379 , 234.33376 ,
       310.9097  , 261.67514 , 257.17688 , 292.70667 , 269.86954 ,
       293.54132 , 299.97525 , 273.157   , 303.02264 , 340.29538 ,
       416.10162 , 283.07007 , 360.62222 , 372.32306 , 320.8344  ,
       298.0853  , 281.95743 , 254.29605 , 311.97482 , 277.93137 ,
       261.30914 , 287.5816  , 254.98253 , 257.10098 , 255.18086 ,
       340.8647  , 260.91446 , 330.92484 , 291.1658  , 284.98868 ,
       304.44995 , 304.4309  , 378.646   , 301.69244 , 309.9392  ,
       316.72748 , 352.37607 , 320.5493  , 333.678   , 282.2997  ,
       300.25354 , 342.03622 , 278.63467 , 364.29013 , 278.3256  ,
       269.34277 , 268.77325 , 275.36682 , 296.96075 , 284.5537  ,
       294.39203 , 298.59485 , 323.97137 , 309.03134 , 291.2512  ,
       376.08765 , 267.08325 , 276.48373 , 284.29254 , 284.9734  ,
       271.65268 , 264.9646  , 255.07492 , 257.19788 , 259.2903  ,
       243.9535  , 255.83003 , 249.82039 , 333.5513  , 428.84332 ,
       354.00873 , 397.18423 , 275.76355 , 261.72864 , 254.27022 ,
       243.37039 , 266.51962 , 335.2047  , 327.97412 , 282.68448 ,
       274.95154 , 265.28604 , 271.92117 , 276.19162 , 245.52614 ,
       250.53276 , 225.52112 , 254.15453 , 241.40962 , 221.85645 ,
       237.20473 , 203.40837 , 231.90018 , 248.15646 , 206.89964 ,
       180.16937 , 186.18327 , 199.38379 , 170.66101 , 183.67032 ,
       167.9499  , 171.39702 , 164.24843 , 155.88173 , 153.07202 ,
       156.61806 , 141.65099 , 137.91965 , 146.00153 , 187.71574 ,
       140.97281 , 139.54718 , 129.109   , 128.1222  , 132.89798 ,
       137.86665 , 125.573875, 128.85555 , 146.38857 , 129.43396 ,
       128.87257 , 131.66841 , 114.34183 , 128.83656 , 125.68192 ,
       116.21729 , 123.80484 , 130.76003 , 115.00698 , 175.36699 ,
       135.19774 , 112.45836 , 124.30182 , 163.53741 , 123.439926,
       148.74066 , 163.47751 , 121.01831 , 137.65762 , 159.50792 ,
       180.11537 , 170.19621 , 215.43076 , 232.60745 , 218.49472 ,
       232.72475 , 216.91974 , 262.35876 , 313.5901  , 249.61707 ,
       306.8658  , 281.4747  , 245.27382 , 279.5458  , 249.99254 ,
       276.37967 , 281.72192 , 228.8153  , 300.35498 , 284.94528 ,
       304.18216 , 264.40994 , 264.1738  , 292.00674 , 283.5702  ,
       305.13956 , 272.52188 , 264.34692 , 305.42123 , 294.9019  ,
       301.6475  , 433.6961  , 261.15216 , 253.20422 , 274.985   ,
       259.93808 , 315.63907 , 313.2918  , 249.87839 , 272.96286 ,
       298.67874 , 260.4483  , 253.11325 , 273.78622 , 347.64072 ,
       361.6027  , 279.88034 , 265.1378  , 273.12997 , 237.81693 ,
       285.64017 , 248.44118 , 359.11258 , 349.07205 , 308.28894 ,
       316.73328 , 304.19998 , 245.34357 , 283.3011  , 282.69675 ,
       316.87683 , 262.55374 , 388.34683 , 314.81293 , 264.83646 ,
       286.91626 , 255.04718 , 288.76068 , 341.65436 , 334.66766 ,
       293.93536 , 294.34982 , 252.45963 , 281.5339  , 282.7292  ,
       243.31384 , 301.54828 , 392.26813 , 322.37238 , 286.8397  ,
       300.92706 , 277.53333 , 310.5054  , 320.99918 , 261.50073 ,
       343.7002  , 247.25351 , 257.3514  , 224.2378  , 226.1827  ,
       196.75832 , 242.32933 , 296.5609  , 242.2168  , 203.19429 ,
       225.25038 , 241.7062  , 240.3873  , 268.10446 , 230.51224 ,
       223.28494 , 189.74945 , 183.47849 , 156.2828  , 183.52837 ,
       184.84451 , 162.34947 , 172.85674 , 180.04456 , 139.13553 ,
       174.47476 , 181.89032 , 153.27176 , 148.55197 , 148.19653 ,
       133.51384 , 150.02887 , 147.04684 , 126.847046, 133.51378 ,
       138.44275 , 131.95576 , 134.73444 , 134.03528 , 124.649376,
       136.01202 , 142.38997 , 116.81768 , 135.76079 , 123.15785 ,
       122.67217 , 115.8912  , 122.15757 , 124.37477 , 113.06441 ,
       127.73125 , 124.28141 , 122.34391 , 116.68067 , 114.25538 ,
       130.6133  , 131.92682 , 141.162   , 150.3272  , 150.48254 ,
       180.61473 , 188.08623 , 205.6266  , 214.5683  , 236.98404 ,
       311.22092 , 290.76318 , 252.02216 , 320.75815 , 378.15024 ,
       481.63614 , 236.54897 , 306.97876 , 265.50717 , 240.77956 ,
       244.28764 , 332.92215 , 287.4732  , 265.2231  , 328.74515 ,
       351.05164 , 287.6309  , 287.1972  , 264.28372 , 246.0992  ,
       287.75766 , 214.45085 , 290.28445 , 243.54144 , 290.48444 ,
       318.83057 , 317.6564  , 304.1382  , 277.057   , 346.3308  ,
       305.12088 , 310.99817 , 372.11374 , 343.71222 , 318.35248 ,
       349.00235 , 282.7766  , 284.4276  , 287.13217 , 285.33786 ,
       366.79623 , 471.84207 , 322.11246 , 355.06183 , 327.44412 ,
       256.07605 , 319.83105 , 311.22647 , 329.5897  , 324.37027 ,
       349.66815 , 398.46793 , 314.37088 , 432.47757 , 302.594   ,
       314.23163 , 309.53268 , 398.65738 , 349.89813 , 269.97116 ,
       270.36865 , 437.94247 , 270.20465 , 271.64587 , 262.5256  ,
       281.98566 , 282.7628  , 479.34012 , 290.75757 , 256.76263 ,
       314.4075  , 250.33992 , 241.90276 , 244.27101 , 238.0547  ,
       259.19794 , 243.68864 , 259.00586 , 240.09196 , 271.41403 ,
       230.01953 , 255.96776 , 272.5123  , 357.7926  , 237.22229 ,
       259.7646  , 268.76517 , 352.962   , 299.17694 , 289.2813  ,
       386.23434 , 294.31244 , 305.71027 , 269.87875 , 246.67398 ,
       407.56305 , 411.93732 , 227.27519 , 226.41719 , 216.9765  ,
       206.00502 , 172.12328 , 170.76006 , 168.86122 , 159.56078 ,
       155.94305 , 209.8086  , 209.89322 , 139.20241 , 155.32324 ,
       143.0515  , 134.49484 , 146.71283 , 132.51454 , 137.10255 ,
       141.42809 , 126.80798 , 131.63129 , 139.48383 , 120.01754 ,
       127.596695, 125.24553 , 115.659546, 117.85732 , 129.40652 ,
       116.901215, 124.42424 , 118.005714, 116.77205 , 125.92656 ,
       122.88261 , 120.25258 , 133.96468 , 139.79643 , 118.83715 ,
       149.04994 , 159.52786 , 157.72572 , 190.77563 , 201.67923 ,
       207.38245 , 213.68677 , 241.51253 , 233.56738 , 284.5839  ,
       293.64737 , 236.379   , 267.9907  , 274.05185 , 292.62985 ,
       314.48172 , 302.01154 , 256.44064 , 306.0366  , 367.58932 ,
       289.8807  , 339.00644 , 336.0655  , 436.3028  , 302.99088 ,
       322.2683  , 308.98477 , 261.81467 , 270.93726 , 262.76883 ,
       301.02515 , 294.69556 , 296.89966 , 275.19812 , 268.5367  ,
       307.07147 , 410.77597 , 391.51477 , 333.90848 , 303.22382 ,
       264.06445 , 278.45435 , 331.50525 , 307.63733 , 292.5986  ,
       351.10184 , 361.8632  , 295.46    , 271.98184 , 346.3101  ,
       337.8729  , 421.75192 , 330.2415  , 369.09384 , 337.26797 ,
       268.156   , 306.11856 , 330.17282 , 304.56314 , 334.786   ,
       314.61487 , 326.90393 , 327.26788 , 301.6828  , 365.11905 ,
       274.6524  , 258.64758 , 311.63162 , 299.7862  , 229.50012 ,
       277.43204 , 273.37726 , 274.90924 , 275.07623 , 329.82498 ,
       294.834   , 264.24683 , 261.55197 , 253.93796 , 263.39355 ,
       272.55374 , 245.15137 , 247.00171 , 226.05893 , 378.10208 ,
       235.85654 , 203.54546 , 240.92355 , 255.91347 , 240.96667 ,
       219.77455 , 236.38498 , 209.40576 , 249.92435 , 288.87704 ,
       257.19788 , 298.59235 , 328.44818 , 259.0231  , 232.10968 ,
       287.0427  , 268.86017 , 316.82578 , 269.3911  };

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
    static float weight_array[14];

    // Loading data for tests if first time booting
    if (first_send_message) {
            previously_transmitted = conso_data[0];//load_data_Init();
            first_send_message = false;
            predict_nok = true;
            index_value = 0;
            skipped = 0;

    // Weight array initialisation
    // Probably will need refactor between variables (y_t and c_t) and statics
    weight_array[0] = -0.218822;
    weight_array[1] = -0.011803;
    weight_array[2] = 1.1231076;
    weight_array[3] =0.4633199;
    weight_array[4] = -1.190576;
    weight_array[5] = -0.0426245;
    weight_array[6] =0.45407116;
    weight_array[7] =0.00370111;
    weight_array[8] = 0.3802879;
    weight_array[9] = 0.8057212;
    weight_array[10] = 0.18706907;
    weight_array[11] =  1.;
    weight_array[12] = -0.22962211;
    weight_array[13] =0.14464839;
    }

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

    float y_t_1 = weight_array[0];
    float c_t_1 = weight_array[1];

    // Dual prediction
    // We position ourselves as the server and base prediction on previously transmitted value

    // Quantize the input from floating-point to integer
    // int8_t x_quantized = x_val / input->params.scale + input->params.zero_point;
    // Place the quantized input in the model's input tensor
    // input->data.int8[0] = x_quantized;

    float y_val = complete_nn(x_val, weight_array);
    
    // Obtain the quantized output from model's output tensor
    // int8_t y_quantized = output->data.int8[0];
    // Dequantize the output from integer to floating-point
    // float y_val = (y_quantized - output->params.zero_point) * output->params.scale;

    // Determining difference unscaled
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
    

        // 2.5 Retro fitting (fits c_t and y_t from nn to match transmitted data

        retro_fitting(y_t_1, c_t_1, weight_array, x_val, (y_val - previously_transmitted - x_min) / (x_max - x_min) * (tx_max - tx_min) + tx_min);
        printf("y_t %f\n",weight_array[0]);
        printf("c_t %f\n",weight_array[1]);
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

void retro_fitting(float y_t_1, float c_t_1, float * weight_array, float x_val, float y_val) {

    float dense_W = -1.2636627;
    float dense_b = 0.17336495;

    struct LSTMCell lstmCellinChain{};

    lstmCellinChain.y_t_1 = y_t_1;
    lstmCellinChain.c_t_1 = c_t_1;

    lstmCellinChain.W_i_x = weight_array[2];
    lstmCellinChain.W_f_x = weight_array[3];
    lstmCellinChain.W_z_x = weight_array[4];
    lstmCellinChain.W_o_x = weight_array[5];

    lstmCellinChain.W_i_y = weight_array[6];
    lstmCellinChain.W_f_y = weight_array[7];
    lstmCellinChain.W_z_y = weight_array[8];
    lstmCellinChain.W_o_y = weight_array[9];

    lstmCellinChain.b_i = weight_array[10];
    lstmCellinChain.b_f = weight_array[11];
    lstmCellinChain.b_z = weight_array[12];
    lstmCellinChain.b_o = weight_array[13];

    float y_t = (y_val - dense_b) / dense_W;

    float z_t = 0.;
    float i_t = 0.;
    float f_t = 0.;
    float o_t = 0.;

    float c_t = 0;

    // Forget Gate
    f_t = sigmoid_function(lstmCellinChain.W_f_x * x_val + lstmCellinChain.W_f_y * lstmCellinChain.y_t_1 + lstmCellinChain.b_f);

    // Input Gate
    i_t = sigmoid_function(lstmCellinChain.W_i_x * x_val + lstmCellinChain.W_i_y * lstmCellinChain.y_t_1 + lstmCellinChain.b_i);

    // Current cell state
    z_t = g(lstmCellinChain.W_z_x * x_val + lstmCellinChain.W_z_y * lstmCellinChain.y_t_1 + lstmCellinChain.b_z);

    // Output Cell state
    o_t = sigmoid_function(lstmCellinChain.W_o_x * x_val + lstmCellinChain.W_o_y * lstmCellinChain.y_t_1 + lstmCellinChain.b_o);

    c_t = (tanh_minus_1(y_t/o_t)-(z_t*i_t))/f_t;

    weight_array[0] = y_t;
    weight_array[1] = c_t;
    
    return;
}

float sigmoid_function (float input) {
    return 1/(1+(exp(-input)));
}

float dot( float a, float b) {
    return a*b;
}

float g(float input) {
    return tanh(input);
}

float h(float input) {
    return tanh(input);
}

float tanh_minus_1 (float input) {
    return (0.5 * log((1+input)/(1-input)));
}
//EOF
