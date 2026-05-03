#include <Wire.h>

#include <Adafruit_MotorShield.h>


// --- PHYSICAL CONFIGURATION ---

const double MOTOR_ENCODER_PPR = 12.0;
S
const double GEAR_RATIO = 80;

const double TRACK_WIDTH_CM = 12.7; 

const double TOTAL_PPR = MOTOR_ENCODER_PPR * GEAR_RATIO;


const double WHEEL_DIAMETER_INCH = 2.25;

const double CM_PER_INCH = 2.54;

const double CIRCUMFERENCE_CM = WHEEL_DIAMETER_INCH * CM_PER_INCH * PI;

const double CM_PER_TICK = CIRCUMFERENCE_CM / TOTAL_PPR;


// --- PID TUNING ---

double FF = 2.5; // Fudge-Factor for Straight Lines

double FFT = 2; // Fudge-Factor for Turning

double FFV = 1; // Fudge-Factor for Velocity Mode

double Kp = 30, Ki = 125, Kd = 0.2875;


const int ENCA[] = {3, 4, 5, 6};

const int ENCB[] = {8, 9, 10, 11};


Adafruit_MotorShield AFMS = Adafruit_MotorShield();

Adafruit_DCMotor * motors[4];


volatile long counts[4] = {0, 0, 0, 0};

long prevCounts[4] = {0, 0, 0, 0};

long startCounts[4] = {0, 0, 0, 0};

double targetRPM[4] = {0, 0, 0, 0};

double prevTargetRPM[4] = {0, 0, 0, 0};

double currentRPM[4] = {0, 0, 0, 0}; 

double lastError[4] = {0, 0, 0, 0};

double integral[4] = {0, 0, 0, 0};


bool isMoving = false;

double targetDistanceCm = 0;

unsigned long prevPIDTime = 0;

const int PID_INTERVAL = 50;

const unsigned long VELOCITY_TIMEOUT_MS = 500;

const double MECANUM_OMEGA_K = TRACK_WIDTH_CM * PI / 180.0;

// --- DIRECTION / CONTROL CALIBRATION ---
// Set to -1 if +Y command currently moves right instead of left.
const int VY_SIGN = -1;

// Per-wheel sign calibration (index map: 0=FR, 1=FL, 2=RR, 3=RL).
const int MOTOR_DIR[4] = {1, 1, 1, 1};
const int ENC_DIR[4] = {1, 1, 1, 1};

char inputBuffer[64];

int bufferIdx = 0;

double cmdVxCmS = 0;

double cmdVyCmS = 0;

double cmdOmegaDegS = 0;

unsigned long lastVelocityCmdTime = 0;

bool velocityModeActive = false;


// --- ISRs ---

void isrM1() { (digitalRead(ENCB[0])) ? counts[0]-- : counts[0]++; }

void isrM2() { (digitalRead(ENCB[1])) ? counts[1]++ : counts[1]--; }

void isrM3() { (digitalRead(ENCB[2])) ? counts[2]-- : counts[2]++; }

void isrM4() { (digitalRead(ENCB[3])) ? counts[3]++ : counts[3]--; }


// --- HELPER FUNCTIONS ---

void stopRobot() {

    isMoving = false;

    for(int i=0; i<4; i++) { 

        targetRPM[i] = 0; 

        prevTargetRPM[i] = 0;

        integral[i] = 0; 

    }

}


void resetEncoders() {

    noInterrupts();

    for(int j=0; j<4; j++) startCounts[j] = counts[j];

    interrupts();

}


/*
 * Applies the latest streamed body-frame velocity setpoint to wheel RPM targets
 * using mecanum forward kinematics and the robot's wheel geometry.
 */
void applyVelocitySetpointToRPM() {

    double vy = (double)VY_SIGN * cmdVyCmS;

    double omegaTerm = cmdOmegaDegS * MECANUM_OMEGA_K;

    double flCmS = cmdVxCmS - vy - omegaTerm;

    double frCmS = cmdVxCmS + vy + omegaTerm;

    double rlCmS = cmdVxCmS + vy - omegaTerm;

    double rrCmS = cmdVxCmS - vy + omegaTerm;


    // Index map: 0=FR, 1=FL, 2=RR, 3=RL.
    targetRPM[0] = (frCmS / CIRCUMFERENCE_CM) * 60.0; // Front Right (FR)

    targetRPM[1] = (flCmS / CIRCUMFERENCE_CM) * 60.0 * FFV; // Front Left (FL)

    targetRPM[2] = (rrCmS / CIRCUMFERENCE_CM) * 60.0; // Rear Right (RR)

    targetRPM[3] = (rlCmS / CIRCUMFERENCE_CM) * 60.0 * FFV; // Rear Left  (RL)

    isMoving = false;

    targetDistanceCm = 0;

}


/*
 * Stops all wheel outputs immediately and clears streamed velocity state.
 */
void immediateStopAndClearVelocity() {

    stopRobot();

    cmdVxCmS = 0;

    cmdVyCmS = 0;

    cmdOmegaDegS = 0;

    velocityModeActive = false;

    for (int i = 0; i < 4; i++) {

        motors[i]->setSpeed(0);

        motors[i]->run(RELEASE);

    }

}


/*
 * Parses a streamed velocity command line in the form:
 * V,<vx_cm/s>,<vy_cm/s>,<omega_deg/s>
 */
bool parseVelocityCommand(char* line) {

    if (toupper(line[0]) != 'V') {

        return false;

    }


    char* p1 = strchr(line, ',');

    if (!p1) {

        return false;

    }

    char* p2 = strchr(p1 + 1, ',');

    if (!p2) {

        return false;

    }

    char* p3 = strchr(p2 + 1, ',');

    if (!p3) {

        return false;

    }


    cmdVxCmS = atof(p1 + 1);

    cmdVyCmS = atof(p2 + 1);

    cmdOmegaDegS = atof(p3 + 1);

    lastVelocityCmdTime = millis();

    velocityModeActive = true;

    applyVelocitySetpointToRPM();

    return true;

}


/*
 * Runs the legacy F/T/S parser exactly as before for non-streaming commands.
 */
void processLegacyCommand(char* line) {

    char type = ' ';

    double val1 = 0, val2 = 0;


    char* startPtr = strpbrk(line, "FTSfts");
    if (startPtr) {

        type = toupper(startPtr[0]);

        val1 = atof(startPtr + 1);


        char* commaPtr = strchr(startPtr, ',');

        if (commaPtr) {

            val2 = atof(commaPtr + 1);

        }

    }


    if (type == 'F') {

        double rpm = (val1 / CIRCUMFERENCE_CM) * 60.0;

        targetDistanceCm = val2;

        targetRPM[0] = rpm; // Right Front

        targetRPM[1] = rpm * FF; // Left Front

        targetRPM[2] = rpm; // Right Back

        targetRPM[3] = rpm * FF; // Left Back


        isMoving = true;

        resetEncoders();

    }
    else if (type == 'T') {

        double wheelSpeedCmS = (val1 * PI / 180.0) * (TRACK_WIDTH_CM);

        double rpm = (wheelSpeedCmS / CIRCUMFERENCE_CM) * 60.0;

        targetDistanceCm = abs((val2 * PI / 180.0) * (TRACK_WIDTH_CM));


        targetRPM[0] = -rpm; // Right Front

        targetRPM[1] =  rpm * FFT; // Left Front

        targetRPM[2] = -rpm; // Right Back

        targetRPM[3] =  rpm * FFT; // Left Back

        isMoving = true;

        resetEncoders();

    }
    else if (type == 'S') {

        stopRobot();

    }

}


/*
 * Dispatches one completed serial line to streaming commands first, then legacy.
 */
void processCommandLine(char* line) {

    while (*line == ' ' || *line == '\t') {

        line++;

    }


    if (line[0] == '\0') {

        return;

    }


    char commandType = toupper(line[0]);

    if (commandType == 'V') {

        if (!parseVelocityCommand(line)) {

            Serial.println("ERR,V");

        }

        return;

    }

    if (commandType == 'X') {

        immediateStopAndClearVelocity();

        return;

    }


    velocityModeActive = false;

    processLegacyCommand(line);

}


void setup() {

    Serial.begin(115200);

    AFMS.begin();


    for (int i = 0; i < 4; i++) {

        motors[i] = AFMS.getMotor(i + 1);

        pinMode(ENCA[i], INPUT_PULLUP);

        pinMode(ENCB[i], INPUT_PULLUP);

    }

    attachInterrupt(digitalPinToInterrupt(ENCA[0]), isrM1, RISING);

    attachInterrupt(digitalPinToInterrupt(ENCA[1]), isrM2, RISING);

    attachInterrupt(digitalPinToInterrupt(ENCA[2]), isrM3, RISING);

    attachInterrupt(digitalPinToInterrupt(ENCA[3]), isrM4, RISING);

    prevPIDTime = millis();

}


void loop() {

    unsigned long currentTime = millis();


    // --- 1. PID & MOTOR CONTROL SECTION ---

    if (currentTime - prevPIDTime >= PID_INTERVAL) {

        unsigned long elapsedMs = currentTime - prevPIDTime;

        double dt = elapsedMs / 1000.0;

        double deltaTicksLoop[4] = {0, 0, 0, 0};

        prevPIDTime = currentTime;


        if (velocityModeActive) {

            // Continuous velocity-hold behavior:
            // keep applying the last commanded V setpoint until a new V
            // arrives or an explicit stop command (X/S) is received.
            applyVelocitySetpointToRPM();

        }


        if (isMoving && targetDistanceCm > 0) {

            noInterrupts();

            long currentProgress = abs(counts[0] - startCounts[0]);

            interrupts();

            if ((currentProgress * CM_PER_TICK) >= targetDistanceCm) {

                stopRobot();

            }

        }


        for (int i = 0; i < 4; i++) {

            noInterrupts();

            long currentCount = counts[i];

            interrupts();


            double deltaTicksRaw = currentCount - prevCounts[i];

            double deltaTicks = (double)ENC_DIR[i] * deltaTicksRaw;

            deltaTicksLoop[i] = deltaTicks;

            currentRPM[i] = (deltaTicks / TOTAL_PPR) / (dt / 60.0);

            prevCounts[i] = currentCount;


            // Reduce stored integral when setpoint drops sharply or changes sign.
            if ((targetRPM[i] * prevTargetRPM[i] < 0) ||
                (abs(targetRPM[i]) < (abs(prevTargetRPM[i]) - 1.0))) {

                integral[i] *= 0.25;

            }


            double error = targetRPM[i] - currentRPM[i];

            double candidateIntegral = constrain(integral[i] + (error * dt), -50, 50);

            double derivative = (error - lastError[i]) / dt;

            double outputNoSat = (Kp * error) + (Ki * candidateIntegral) + (Kd * derivative);

            int pwmNoSat = constrain(abs((int)outputNoSat), 0, 255);

            bool outputSaturated = (pwmNoSat >= 255);

            // Conditional-integration anti-windup.
            if (!outputSaturated ||
                (outputNoSat > 0 && error < 0) ||
                (outputNoSat < 0 && error > 0)) {

                integral[i] = candidateIntegral;

            }

            double output = (Kp * error) + (Ki * integral[i]) + (Kd * derivative);

            lastError[i] = error;

            prevTargetRPM[i] = targetRPM[i];


            // If target speed is zero, release immediately. This avoids
            // wheel self-ramping after a forced manual spin at rest.
            if (abs(targetRPM[i]) < 0.1) {

                motors[i]->setSpeed(0);

                motors[i]->run(RELEASE);

                integral[i] = 0;

            } else {

                motors[i]->setSpeed(constrain(abs((int)output), 0, 255));

                double driveOutput = output * MOTOR_DIR[i];

                motors[i]->run(driveOutput > 0 ? FORWARD : BACKWARD);

            }

        }


        long correctedCounts[4] = {0, 0, 0, 0};

        noInterrupts();

        for (int i = 0; i < 4; i++) {

            correctedCounts[i] = (long)ENC_DIR[i] * counts[i];

        }

        interrupts();


        Serial.print("BR:");

        Serial.print(correctedCounts[0]); // Back Right Motor

        Serial.print(',');

        Serial.print("RL:");

        Serial.print(correctedCounts[1]); // Rear Left Motor

        Serial.print(',');

        Serial.print("FR:");

        Serial.print(correctedCounts[2]); // Front Right Motor

        Serial.print(',');

        Serial.print("FL:");

        Serial.println(correctedCounts[3]); // Front Left Motor

    }


    // --- 2. COMMAND PARSER SECTION ---

    while (Serial.available() > 0) {

        char c = Serial.read();

        

        // Handle end of command

        if (c == '\n' || c == '\r' || c == ')') { 

            inputBuffer[bufferIdx] = '\0';

            

            if (bufferIdx > 0) {

                processCommandLine(inputBuffer);

            }

            bufferIdx = 0; // Reset for next command

        } 

        // Fill buffer (ignoring opening parenthesis if used)

        else if (c != '(' && bufferIdx < 63) {

            inputBuffer[bufferIdx++] = c;

        }

    }

}