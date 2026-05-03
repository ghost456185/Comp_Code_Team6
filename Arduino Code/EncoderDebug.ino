/*
 * Encoder Debug Sketch
 * Board: Arduino Nano Every
 * Purpose: Read and display encoder pin values to check for broken encoders
 */

// Encoder pins (Channel A=interrupt capable, Channel B=regular input) {Correct as of May 3rd, 2026}
const int ENC_FR_B = 5, ENC_FR_A = 10;
const int ENC_FL_B = 11,  ENC_FL_A = 6;
const int ENC_RR_B = 3,  ENC_RR_A = 8;
const int ENC_RL_B = 9,  ENC_RL_A = 4;

// Encoder counters (incremented on A pin CHANGE)
volatile long countFR = 0;
volatile long countFL = 0;
volatile long countRR = 0;
volatile long countRL = 0;

// Timing for output
unsigned long lastPrintTime = 0;
const unsigned long PRINT_INTERVAL_MS = 500;  // Print every 500ms

// ========================================================================
// ENCODER ISRs - Quadrature decoding (A rising/falling, check B state)
// ========================================================================
void isrFR() {
  countFR += (digitalRead(ENC_FR_B) == digitalRead(ENC_FR_A)) ? 1 : -1;
}

void isrFL() {
  countFL += (digitalRead(ENC_FL_B) == digitalRead(ENC_FL_A)) ? 1 : -1;
}

void isrRR() {
  countRR += (digitalRead(ENC_RR_B) == digitalRead(ENC_RR_A)) ? 1 : -1;
}

void isrRL() {
  countRL += (digitalRead(ENC_RL_B) == digitalRead(ENC_RL_A)) ? 1 : -1;
}

// ========================================================================
// SETUP
// ========================================================================
void setup() {
  Serial.begin(115200);
  delay(100);
  
  Serial.println(F("=== ENCODER DEBUG SKETCH ==="));
  Serial.println(F("Reading encoder pin states..."));
  Serial.println();
  
  // Configure all encoder pins as inputs
  pinMode(ENC_FR_A, INPUT_PULLUP);
  pinMode(ENC_FR_B, INPUT_PULLUP);
  pinMode(ENC_FL_A, INPUT_PULLUP);
  pinMode(ENC_FL_B, INPUT_PULLUP);
  pinMode(ENC_RR_A, INPUT_PULLUP);
  pinMode(ENC_RR_B, INPUT_PULLUP);
  pinMode(ENC_RL_A, INPUT_PULLUP);
  pinMode(ENC_RL_B, INPUT_PULLUP);
  
  // Attach interrupts to Channel A pins
  attachInterrupt(digitalPinToInterrupt(ENC_FR_A), isrFR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_FL_A), isrFL, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_RR_A), isrRR, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_RL_A), isrRL, CHANGE);
  
  Serial.println(F("Encoder pins configured. Rotate wheels to test."));
  Serial.println();
  
  lastPrintTime = millis();
}

// ========================================================================
// MAIN LOOP
// ========================================================================
void loop() {
  unsigned long currentTime = millis();
  
  // Print encoder states every PRINT_INTERVAL_MS
  if (currentTime - lastPrintTime >= PRINT_INTERVAL_MS) {
    lastPrintTime = currentTime;
    printEncoderStatus();
  }
}

// ========================================================================
// PRINT ENCODER STATUS
// ========================================================================
void printEncoderStatus() {
  // Read pin states
  int fr_a = digitalRead(ENC_FR_A);
  int fr_b = digitalRead(ENC_FR_B);
  int fl_a = digitalRead(ENC_FL_A);
  int fl_b = digitalRead(ENC_FL_B);
  int rr_a = digitalRead(ENC_RR_A);
  int rr_b = digitalRead(ENC_RR_B);
  int rl_a = digitalRead(ENC_RL_A);
  int rl_b = digitalRead(ENC_RL_B);
  
  // Snapshot counters
  noInterrupts();
  long snap_FR = countFR;
  long snap_FL = countFL;
  long snap_RR = countRR;
  long snap_RL = countRL;
  interrupts();
  
  // Print header
  Serial.println(F("========================================"));
  
  // Front Right
  Serial.print(F("FR (A:6, B:11)  => A="));
  Serial.print(fr_a);
  Serial.print(F(" B="));
  Serial.print(fr_b);
  Serial.print(F("  Count: "));
  Serial.println(snap_FR);
  
  // Front Left
  Serial.print(F("FL (A:10, B:5)  => A="));
  Serial.print(fl_a);
  Serial.print(F(" B="));
  Serial.print(fl_b);
  Serial.print(F("  Count: "));
  Serial.println(snap_FL);
  
  // Rear Right
  Serial.print(F("RR (A:8, B:3)   => A="));
  Serial.print(rr_a);
  Serial.print(F(" B="));
  Serial.print(rr_b);
  Serial.print(F("  Count: "));
  Serial.println(snap_RR);
  
  // Rear Left
  Serial.print(F("RL (A:4, B:9)   => A="));
  Serial.print(rl_a);
  Serial.print(F(" B="));
  Serial.print(rl_b);
  Serial.print(F("  Count: "));
  Serial.println(snap_RL);
  
  Serial.println();
}
