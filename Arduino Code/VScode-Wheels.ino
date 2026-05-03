/*
 * Mecanum Base Closed-Loop Control - Compact Version
 * Board: Arduino Nano Every | Motor Shield: Adafruit V2
 * Coordinate System: +X forward, +Y left, +Z up (X-configuration)
 */

#include <Wire.h>
#include <Adafruit_MotorShield.h>

// ========================================================================
// CONFIGURATION
// ========================================================================
// Encoder pins (Channel A=interrupt, Channel B=sampled)
const int ENC_FR_B = 11,  ENC_FL_B = 10,  ENC_RR_B = 8,  ENC_RL_B = 9;
const int ENC_FR_A = 6,  ENC_FL_A = 5,  ENC_RR_A = 3,   ENC_RL_A = 4;

// Encoder specs
const int ENCODER_PPR = 8;
const int GEARBOX_RATIO = 120;
const int COUNTS_PER_REV = ENCODER_PPR * 2 * GEARBOX_RATIO;  // 1920


// Robot geometry (cm)
const float WHEEL_RADIUS = 3.0;
const float WHEELBASE_X = 13.1;
const float WHEELBASE_Y = 11.7;

// Motor ports and polarity
const int MOTOR_FL_PORT = 4, MOTOR_FR_PORT = 3, MOTOR_RL_PORT = 2, MOTOR_RR_PORT = 1;
const int MOTOR_FL_DIR = 1, MOTOR_FR_DIR = 1, MOTOR_RL_DIR = 1, MOTOR_RR_DIR = 1;

// PI controller (Change Values as Needed)
const float KP = 10.0, KI = 5;
const unsigned long LOOP_PERIOD_MS = 100;  // 10Hz

// Queue and tolerances
const int QUEUE_DEPTH = 5;
const int MAX_LINE_LENGTH = 80;
const int COUNT_TOLERANCE = 10;
const float ANGLE_TOLERANCE = 2.0;

// Turn calibration (compensates for wheel slip during rotation)
// Applied to both speed and encoder targets to maintain constant execution time
const float TURN_CALIBRATION_FACTOR = 1.45;  // Tune: 1.0 = no compensation, >1.0 = faster/longer turns

// Odometry yaw correction: ideal FK over-reports yaw by the slip factor above
const float ODOM_YAW_SCALE = 1.0 / TURN_CALIBRATION_FACTOR;

// ========================================================================
// GLOBALS
// ========================================================================
Adafruit_MotorShield AFMS = Adafruit_MotorShield();
Adafruit_DCMotor *motorFL, *motorFR, *motorRL, *motorRR;

// Encoders
volatile long countFL = 0, countFR = 0, countRL = 0, countRR = 0;
long prevCountFL = 0, prevCountFR = 0, prevCountRL = 0, prevCountRR = 0;
long lastDeltaFL = 0, lastDeltaFR = 0, lastDeltaRL = 0, lastDeltaRR = 0;

// Control
float measuredSpeedFL = 0, measuredSpeedFR = 0, measuredSpeedRL = 0, measuredSpeedRR = 0;
float setpointFL = 0, setpointFR = 0, setpointRL = 0, setpointRR = 0;
float integralFL = 0, integralFR = 0, integralRL = 0, integralRR = 0;

// Dead reckoning
long targetFL = 0, targetFR = 0, targetRL = 0, targetRR = 0;
float targetYaw = 0, currentYaw = 0;
bool isOrbitCommand = false;

// State machine
enum State { WAITING, RUNNING };
State currentState = WAITING;
enum CmdType { CMD_FB, CMD_TURN, CMD_STRAFE, CMD_ORBIT, CMD_VEL, CMD_INVALID };
struct Command { CmdType type; float param1, param2, param3, param4; };

// Velocity-mode state (CMD_VEL runs until superseded, stopped via X, or timeout)
bool isVelCommand = false;
unsigned long velCmdEndMs = 0;  // 0 = no timeout

// Command queue
Command cmdQueue[QUEUE_DEPTH];
int queueHead = 0, queueTail = 0, queueCount = 0;

// Serial & timing
char inputBuffer[MAX_LINE_LENGTH];
int bufferIndex = 0;
unsigned long lastLoopTime = 0;

// ========================================================================
// ENCODER ISRs
// ========================================================================
void isrFL() { countFL += (digitalRead(ENC_FL_B) == digitalRead(ENC_FL_A)) ? 1 : -1; }
void isrFR() { countFR += (digitalRead(ENC_FR_B) == digitalRead(ENC_FR_A)) ? 1 : -1; }
void isrRL() { countRL += (digitalRead(ENC_RL_B) == digitalRead(ENC_RL_A)) ? 1 : -1; }
void isrRR() { countRR += (digitalRead(ENC_RR_B) == digitalRead(ENC_RR_A)) ? 1 : -1; }

// ========================================================================
// SETUP
// ========================================================================
void setup() {
  Serial.begin(115200);
  Serial.println(F("Mecanum Init"));
  
  // Encoder pins
  pinMode(ENC_FL_A, INPUT_PULLUP); pinMode(ENC_FL_B, INPUT_PULLUP);
  pinMode(ENC_FR_A, INPUT_PULLUP); pinMode(ENC_FR_B, INPUT_PULLUP);
  pinMode(ENC_RL_A, INPUT_PULLUP); pinMode(ENC_RL_B, INPUT_PULLUP);
  pinMode(ENC_RR_A, INPUT_PULLUP); pinMode(ENC_RR_B, INPUT_PULLUP);
  
  // Attach interrupts
  attachInterrupt(digitalPinToInterrupt(ENC_FL_A), isrFL, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_FR_A), isrFR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_RL_A), isrRL, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_RR_A), isrRR, CHANGE);
  
  // Motor shield
  if (!AFMS.begin()) { Serial.println(F("ERROR: Motor Shield")); while(1); }
  motorFL = AFMS.getMotor(MOTOR_FL_PORT);
  motorFR = AFMS.getMotor(MOTOR_FR_PORT);
  motorRL = AFMS.getMotor(MOTOR_RL_PORT);
  motorRR = AFMS.getMotor(MOTOR_RR_PORT);
  
  stopAllMotors();
  lastLoopTime = millis();
  Serial.println(F("Ready"));
}

// ========================================================================
// MAIN LOOP
// ========================================================================
void loop() {
  unsigned long currentTime = millis();
  serviceSerial();
  
  if (currentTime - lastLoopTime >= LOOP_PERIOD_MS) {
    unsigned long dt_ms = currentTime - lastLoopTime;
    float dt = dt_ms / 1000.0;
    lastLoopTime = currentTime;
    updateMeasuredSpeeds(dt);
    emitOdometry(dt_ms);

    switch (currentState) {
      case WAITING:
        if (queueCount > 0) startCommand(cmdQueue[queueHead]);
        break;
      case RUNNING:
        updateControl(dt);
        if (isCommandComplete()) completeCommand();
        break;
    }
  }
}

// Emit body-frame odometry deltas every control tick. Forward kinematics
// inverts the IK matrix at lines near setpoint assignment.
void emitOdometry(unsigned long dt_ms) {
  float tFL = (lastDeltaFL / (float)COUNTS_PER_REV) * 2.0 * PI;
  float tFR = (lastDeltaFR / (float)COUNTS_PER_REV) * 2.0 * PI;
  float tRL = (lastDeltaRL / (float)COUNTS_PER_REV) * 2.0 * PI;
  float tRR = (lastDeltaRR / (float)COUNTS_PER_REV) * 2.0 * PI;
  float dx = (WHEEL_RADIUS / 4.0) * (tFL + tFR + tRL + tRR);
  float dy = (WHEEL_RADIUS / 4.0) * (-tFL + tFR + tRL - tRR);
  float dyaw_rad = ODOM_YAW_SCALE * (WHEEL_RADIUS / (2.0 * (WHEELBASE_X + WHEELBASE_Y))) *
                   (-tFL + tFR - tRL + tRR);
  Serial.print(F("O,"));
  Serial.print(dyaw_rad * 180.0 / PI, 4); Serial.print(',');
  Serial.print(dx, 4); Serial.print(',');
  Serial.print(dy, 4); Serial.print(',');
  Serial.println(dt_ms);
}

// ========================================================================
// SERIAL HANDLING
// ========================================================================
void serviceSerial() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      inputBuffer[bufferIndex] = '\0';
      processCommand(inputBuffer);
      bufferIndex = 0;
    } else if (c != '\r' && bufferIndex < MAX_LINE_LENGTH - 1) {
      inputBuffer[bufferIndex++] = c;
    }
  }
}

void processCommand(char* line) {
  Command cmd;
  cmd.type = CMD_INVALID;
  
  char* token = strtok(line, ",");
  if (!token) return;
  
  // Emergency stop
  if (strcmp(token, "X") == 0) {
    queueHead = queueTail = queueCount = 0;
    stopAllMotors();
    isVelCommand = false;
    isOrbitCommand = false;
    currentState = WAITING;
    Serial.println(F("STOP"));
    return;
  }

  // Velocity (twist) command: V,vx_cm_s,vy_cm_s,omega_deg_s[,timeout_ms]
  // Preempts queued/running commands. Runs until another V, an X, or timeout.
  if (strcmp(token, "V") == 0) {
    float vx = 0, vy = 0, omega_deg = 0, timeout_ms = 0;
    token = strtok(NULL, ","); if (token) vx = atof(token);
    token = strtok(NULL, ","); if (token) vy = atof(token);
    token = strtok(NULL, ","); if (token) omega_deg = atof(token);
    token = strtok(NULL, ","); if (token) timeout_ms = atof(token);

    float L = WHEELBASE_X, W = WHEELBASE_Y, r = WHEEL_RADIUS;
    float omega = omega_deg * PI / 180.0;
    float newFL = (1.0/r) * (vx - vy - (L+W)/2.0 * omega);
    float newFR = (1.0/r) * (vx + vy + (L+W)/2.0 * omega);
    float newRL = (1.0/r) * (vx + vy - (L+W)/2.0 * omega);
    float newRR = (1.0/r) * (vx - vy + (L+W)/2.0 * omega);

    if (isVelCommand && currentState == RUNNING) {
      // Smooth setpoint update — keep integrators and encoder counters running
      setpointFL = newFL; setpointFR = newFR;
      setpointRL = newRL; setpointRR = newRR;
    } else {
      // Preempt any queued or running non-VEL command
      queueHead = queueTail = queueCount = 0;
      integralFL = integralFR = integralRL = integralRR = 0;
      setpointFL = newFL; setpointFR = newFR;
      setpointRL = newRL; setpointRR = newRR;
      isVelCommand = true;
      isOrbitCommand = false;
      currentState = RUNNING;
    }
    velCmdEndMs = (timeout_ms > 0.0) ? (millis() + (unsigned long)timeout_ms) : 0UL;
    return;
  }

  // Parse command type
  if (strcmp(token, "FB") == 0) cmd.type = CMD_FB;
  else if (strcmp(token, "T") == 0) cmd.type = CMD_TURN;
  else if (strcmp(token, "S") == 0) cmd.type = CMD_STRAFE;
  else if (strcmp(token, "O") == 0) cmd.type = CMD_ORBIT;
  
  // Parse parameters
  if (cmd.type != CMD_INVALID) {
    token = strtok(NULL, ","); if (token) cmd.param1 = atof(token);
    token = strtok(NULL, ","); if (token) cmd.param2 = atof(token);
    if (cmd.type == CMD_ORBIT) {
      token = strtok(NULL, ","); if (token) cmd.param3 = atof(token);
      token = strtok(NULL, ","); if (token) cmd.param4 = atof(token);
    }
    enqueueCommand(cmd);
  }
}

void enqueueCommand(Command cmd) {
  if (queueCount >= QUEUE_DEPTH) return;
  cmdQueue[queueTail] = cmd;
  queueTail = (queueTail + 1) % QUEUE_DEPTH;
  queueCount++;
}

Command dequeueCommand() {
  Command cmd = cmdQueue[queueHead];
  queueHead = (queueHead + 1) % QUEUE_DEPTH;
  queueCount--;
  return cmd;
}

// ========================================================================
// COMMAND EXECUTION
// ========================================================================
void startCommand(Command cmd) {
  // Reset state
  noInterrupts();
  countFL = countFR = countRL = countRR = 0;
  interrupts();
  prevCountFL = prevCountFR = prevCountRL = prevCountRR = 0;
  currentYaw = 0;
  integralFL = integralFR = integralRL = integralRR = 0;
  
  float Vx = 0, Vy = 0, omega = 0;
  isOrbitCommand = false;
  
  switch (cmd.type) {
    case CMD_FB: {
      Vx = cmd.param1;
      float wheelTravel = cmd.param2 * (cmd.param1 >= 0 ? 1 : -1);
      long wheelCounts = (long)((wheelTravel / WHEEL_RADIUS) * COUNTS_PER_REV / (2.0 * PI));
      targetFL = targetFR = targetRL = targetRR = wheelCounts;
      break;
    }
    
    case CMD_STRAFE: {
      Vy = cmd.param1;
      float wheelTravel = cmd.param2 * (cmd.param1 >= 0 ? 1 : -1);
      long wheelCounts = (long)((wheelTravel / WHEEL_RADIUS) * COUNTS_PER_REV / (2.0 * PI));
      targetFL = -wheelCounts; targetFR = wheelCounts;
      targetRL = wheelCounts;  targetRR = -wheelCounts;
      break;
    }
    
    case CMD_TURN: {
      omega = cmd.param1 * PI / 180.0 * TURN_CALIBRATION_FACTOR;  // Apply calibration to speed
      float deltaTheta = cmd.param2 * (cmd.param1 >= 0 ? 1 : -1) * PI / 180.0;
      float wheelDist = sqrt((WHEELBASE_X/2)*(WHEELBASE_X/2) + (WHEELBASE_Y/2)*(WHEELBASE_Y/2));
      float arcLength = wheelDist * fabs(deltaTheta);
      long wheelCounts = (long)((arcLength / WHEEL_RADIUS) * COUNTS_PER_REV / (2.0 * PI) * TURN_CALIBRATION_FACTOR);  // Apply calibration to targets
      if (deltaTheta < 0) wheelCounts = -wheelCounts;
      targetFL = -wheelCounts; targetFR = wheelCounts;
      targetRL = -wheelCounts; targetRR = wheelCounts;
      break;
    }
    
    case CMD_ORBIT: {
      float R = sqrt(cmd.param3*cmd.param3 + cmd.param4*cmd.param4);
      float sgn = (cmd.param2 >= 0) ? 1.0 : -1.0;
      omega = sgn * cmd.param1 / R;
      Vx = omega * cmd.param4;   // omega × Yo
      Vy = -omega * cmd.param3;  // -omega × Xo
      isOrbitCommand = true;
      targetYaw = cmd.param2;
      break;
    }
    
    default: return;
  }
  
  // Compute wheel speed setpoints
  float L = WHEELBASE_X, W = WHEELBASE_Y, r = WHEEL_RADIUS;
  setpointFL = (1.0/r) * (Vx - Vy - (L+W)/2.0 * omega);
  setpointFR = (1.0/r) * (Vx + Vy + (L+W)/2.0 * omega);
  setpointRL = (1.0/r) * (Vx + Vy - (L+W)/2.0 * omega);
  setpointRR = (1.0/r) * (Vx - Vy + (L+W)/2.0 * omega);
  
  dequeueCommand();
  currentState = RUNNING;
}

void completeCommand() {
  Serial.println(F("Done"));
  stopAllMotors();
  isVelCommand = false;
  isOrbitCommand = false;
  currentState = WAITING;
}

// ========================================================================
// CONTROL
// ========================================================================
void updateMeasuredSpeeds(float dt) {
  long snapFL, snapFR, snapRL, snapRR;
  noInterrupts();
  snapFL = countFL; snapFR = countFR; snapRL = countRL; snapRR = countRR;
  interrupts();
  
  long deltaFL = snapFL - prevCountFL, deltaFR = snapFR - prevCountFR;
  long deltaRL = snapRL - prevCountRL, deltaRR = snapRR - prevCountRR;
  
  lastDeltaFL = deltaFL; lastDeltaFR = deltaFR;
  lastDeltaRL = deltaRL; lastDeltaRR = deltaRR;
  
  prevCountFL = snapFL; prevCountFR = snapFR;
  prevCountRL = snapRL; prevCountRR = snapRR;
  
  // Convert deltas to wheel speeds (rad/s)
  if (dt > 0.001) {
    measuredSpeedFL = (deltaFL / (float)COUNTS_PER_REV) * 2.0 * PI / dt;
    measuredSpeedFR = (deltaFR / (float)COUNTS_PER_REV) * 2.0 * PI / dt;
    measuredSpeedRL = (deltaRL / (float)COUNTS_PER_REV) * 2.0 * PI / dt;
    measuredSpeedRR = (deltaRR / (float)COUNTS_PER_REV) * 2.0 * PI / dt;
  }
}

void updateControl(float dt) {
  // Yaw accumulation for orbit
  if (isOrbitCommand && dt > 0.001) {
    float theta_FL = (lastDeltaFL / (float)COUNTS_PER_REV) * 2.0 * PI;
    float theta_FR = (lastDeltaFR / (float)COUNTS_PER_REV) * 2.0 * PI;
    float theta_RL = (lastDeltaRL / (float)COUNTS_PER_REV) * 2.0 * PI;
    float theta_RR = (lastDeltaRR / (float)COUNTS_PER_REV) * 2.0 * PI;
    float omega = (WHEEL_RADIUS / (2.0*(WHEELBASE_X+WHEELBASE_Y))) * 
                  (-theta_FL + theta_FR - theta_RL + theta_RR) / dt;
    currentYaw += omega * dt * 180.0 / PI;
  }
  
  // PI control
  int pwmFL = computePI(measuredSpeedFL, setpointFL, integralFL, dt);
  int pwmFR = computePI(measuredSpeedFR, setpointFR, integralFR, dt);
  int pwmRL = computePI(measuredSpeedRL, setpointRL, integralRL, dt);
  int pwmRR = computePI(measuredSpeedRR, setpointRR, integralRR, dt);
  
  setMotor(motorFL, pwmFL, MOTOR_FL_DIR);
  setMotor(motorFR, pwmFR, MOTOR_FR_DIR);
  setMotor(motorRL, pwmRL, MOTOR_RL_DIR);
  setMotor(motorRR, pwmRR, MOTOR_RR_DIR);
}

int computePI(float measured, float setpoint, float &integral, float dt) {
  const float MOTOR_MAX_OMEGA = 150.0 * (2.0 * PI / 60.0);  // 150 RPM @ PWM 255
  float feedforward = (setpoint / MOTOR_MAX_OMEGA) * 255.0;
  float error = setpoint - measured;
  integral += error * dt;
  float output = feedforward + KP * error + KI * integral;
  return constrain((int)output, -255, 255);
}

void setMotor(Adafruit_DCMotor* motor, int pwm, int dirPolarity) {
  int adjustedPWM = pwm * dirPolarity;
  if (adjustedPWM > 0) {
    motor->setSpeed(abs(adjustedPWM));
    motor->run(FORWARD);
  } else if (adjustedPWM < 0) {
    motor->setSpeed(abs(adjustedPWM));
    motor->run(BACKWARD);
  } else {
    motor->setSpeed(0);
    motor->run(RELEASE);
  }
}

void stopAllMotors() {
  setpointFL = setpointFR = setpointRL = setpointRR = 0;
  integralFL = integralFR = integralRL = integralRR = 0;
  motorFL->run(RELEASE); motorFR->run(RELEASE);
  motorRL->run(RELEASE); motorRR->run(RELEASE);
}

// ========================================================================
// COMPLETION DETECTION
// ========================================================================
bool isCommandComplete() {
  if (isVelCommand) {
    // VEL commands run until timeout, superseded by another V, or X stop
    return (velCmdEndMs > 0UL) && (millis() >= velCmdEndMs);
  }
  if (isOrbitCommand) {
    float error = fabs(currentYaw - targetYaw);
    if (error < ANGLE_TOLERANCE) return true;
    if (targetYaw >= 0 && currentYaw >= targetYaw) return true;
    if (targetYaw < 0 && currentYaw <= targetYaw) return true;
  } else {
    long snapFL, snapFR, snapRL, snapRR;
    noInterrupts();
    snapFL = countFL; snapFR = countFR; snapRL = countRL; snapRR = countRR;
    interrupts();
    
    if (checkTargetReached(snapFL, targetFL, setpointFL) &&
        checkTargetReached(snapFR, targetFR, setpointFR) &&
        checkTargetReached(snapRL, targetRL, setpointRL) &&
        checkTargetReached(snapRR, targetRR, setpointRR)) {
      return true;
    }
  }
  return false;
}

bool checkTargetReached(long current, long target, float setpoint) {
  long error = abs(current - target);
  if (error < COUNT_TOLERANCE) return true;
  if (setpoint >= 0 && current >= target) return true;
  if (setpoint < 0 && current <= target) return true;
  return false;
}
