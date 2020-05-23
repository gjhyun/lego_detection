// Authors - Nikhil Uplekar, Nathan Koenigsmark, Huy Do
// ENEE408I Fall 2019 Group 2

// Code for slave ardino, takes serial inputs, runs RTOS to multitask

// Order API (no whitespace please. Just the newlines)
//  forward -> f
//  left -> l
//  right -> r
//  backwards -> b
//  halt -> h
//  autoOn -> o
//  autoOff -> z
//  personFollow -> p
//  aprilFollow -> a

#include <Arduino_FreeRTOS.h>
#include "MotorControl.h"
#include "Ultrasound.h"

//constants
byte PWM = 60;
const int STOP_DISTANCE_CENTER = 15; // cm
const int STOP_DISTANCE_SIDE   = 10; // cm

// enum for Direction
enum Directions {forward, backward, left, right, halt};
  // 0 - forward, 1 - backwards, 2 - left, 3 - right, 4 - halt
  
// enum for type of following/avoidance
enum Following {person, april};
  // 0 - person, 1 - april


//globals for motor control
boolean avoidObstaclesEnabled = false; // Default to false
boolean freeRoam = false; // Not supported
Directions currDir = halt;
Directions prevDir = halt;
Following followType = person; // Default to person since it's less obstructive

//globals for crash avoidance
boolean dangerCenter = false;
boolean dangerLeft   = false;
boolean dangerRight  = false;
boolean dangerDetected = false;
long leftDistance = 0, centerDistance = 0, rightDistance = 0;
boolean lastCommandFromSerial = false;

//prototypes for RTOS tasks
void updateOrders  (void *pvParameters);
void updatePingData(void *pvParameters);
void driveACR      (void *pvParameters);

// Ping sensor pins
const int center_ping_pin = 5;
const int left_ping_pin   = 2;
const int right_ping_pin  = 11;

// Motor pins
const int LeftMotor_pin    = 8;
const int RightMotor_pin   = 7;
const int LeftMotorPWM_pin  = 10;
const int RightMotorPWM_pin = 9;

// OOP Initializations
MotorControl leftMotor  (LeftMotor_pin,  LeftMotorPWM_pin);
MotorControl rightMotor (RightMotor_pin, RightMotorPWM_pin);
Ultrasound leftUltrasound   (left_ping_pin);
Ultrasound centerUltrasound (center_ping_pin);
Ultrasound rightUltrasound  (right_ping_pin);

void setup() {
  Serial.begin(9600); // Set baud-rate
//  xTaskCreate(driveACR,       (const portCHAR *) "Driving",         128, NULL, 1, NULL); // Priority 1
    go_forward(); // Guarantee that both motors are not moving at start
  xTaskCreate(updateOrders,   (const portCHAR *) "Updating Orders", 128, NULL, 1, NULL); // Priority 2
  xTaskCreate(updatePingData, (const portCHAR *) "Updating Pings",  128, NULL, 2, NULL); // Priority 3

  set_speed(PWM, PWM); // Kind-of useless since we set it on every movement command (see below)
}

// This is supposed to be empty (lets RTOS run uninterupted)
void loop() {
}

void set_speed(const int left_speed, const int right_speed) {
  leftMotor.setPWM(left_speed);
  rightMotor.setPWM(right_speed); 
}

void go_stop() {
  leftMotor.halt();
  rightMotor.halt();
//  Serial.print(" stop");
}

void go_forward() {
  set_speed(PWM, PWM-3);
  leftMotor.forward();
  rightMotor.forward();
//  Serial.print(" forward");
}

void go_backward() {
  set_speed(PWM, PWM);  // Compensate for difference that occurs when backing up
  leftMotor.backward();
  rightMotor.backward();
//  Serial.print(" backward");
}

void go_left() {
  set_speed(PWM, PWM);
  leftMotor.backward();
  rightMotor.forward();
//  Serial.print(" left");
}

void go_right() {
  set_speed(PWM, PWM);
  leftMotor.forward();
  rightMotor.backward();
//  Serial.print(" right");
}

void respondToCurrDir() {
  // TODO - if lastCommand was not from serial then don't check the currDir != prevDir
  
  // Only need to act on the currDir value if it's different from the prevDir
  if (currDir != prevDir) {
    if (currDir == forward)
      go_forward();
    else if (currDir == left)
      go_left();
    else if (currDir == right)
      go_right();
    else if (currDir == backward)
      go_backward();
    else if (currDir == halt)
      go_stop();
    else
      Serial.println("Error in respondToCurrDir() - currDir not found");
  }
}

//check the orders string for new commands
void updateOrders(void *pvParameters){ 
  while(1) {
    if(dangerDetected == 1){
// this is a test
      if (centerDistance <= STOP_DISTANCE_CENTER) { // If Danger at center...
          // Check whether we should go left or right
          if (leftDistance <= rightDistance) { // Right has more room than left, so go right.
              lastCommandFromSerial = false;  
              prevDir = currDir;
              currDir = right;
              respondToCurrDir();
//              Serial.println("go back right");
          }
          else { // Left has more room than right, so go left.
              lastCommandFromSerial = false;  
              prevDir = currDir;
              currDir = left;
              respondToCurrDir();
//              Serial.println("go back left");
          }
        }
        else if (leftDistance <= STOP_DISTANCE_SIDE) { // Issue @ left, so go right
            lastCommandFromSerial = false;  
            prevDir = currDir;
            currDir = right;
            respondToCurrDir();
//            Serial.println("go right");
        }
        else if (rightDistance <= STOP_DISTANCE_SIDE) { // Issue @ right, so go left
            lastCommandFromSerial = false;  
            prevDir = currDir;
            currDir = left;
            respondToCurrDir();
//            Serial.println("go left");
        }
        else { // No issue - this shouldn't be possible
            Serial.println("error: this shouldn't be possible"); 
        }
    }
    else{
      prevDir = currDir;
      currDir = forward;
      respondToCurrDir();
    }
    
    vTaskDelay(50/ portTICK_PERIOD_MS);
  }
}

//check sensors for new obstacles
void updatePingData(void *pvParameters) {
  while(1) {
    // Get distances
    //Serial.println("left");
    leftDistance   = leftUltrasound.getDistance();
    //Serial.println("center");
    centerDistance = centerUltrasound.getDistance();
    //Serial.println("right");
    rightDistance  = rightUltrasound.getDistance();
    
    // Update danger booleans
    dangerLeft   = leftDistance   <= STOP_DISTANCE_SIDE;
    dangerCenter = centerDistance <= STOP_DISTANCE_CENTER;
    dangerRight  = rightDistance  <= STOP_DISTANCE_SIDE;
    dangerDetected = dangerCenter || dangerRight || dangerLeft;
    Serial.println(dangerDetected);
    vTaskDelay(150 / portTICK_PERIOD_MS);
  }
}
