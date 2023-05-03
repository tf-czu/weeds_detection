String inputString = "";
bool stringComplete = false;
long delayVal = 0;

void setup() {
  Serial.begin(115200);
  Serial.println("ready\n");
  // reserve 50 bytes for the inputString:
  inputString.reserve(50);
  pinMode(4, OUTPUT);  // LED
  pinMode(5, OUTPUT);  // camera trigger

}

void loop() {
  if (stringComplete) {
    // Serial.println(inputString);
    delayVal = inputString.toInt();
    if (delayVal == 0) {
      delayVal = 1000;  // default time in us
      Serial.println(0);
    }
    else if (delayVal > 160) {
      delayVal = 16000;
      Serial.println(delayVal);
    }
    else {
      delayVal = delayVal*100;
      Serial.println(delayVal);
    }
    digitalWrite(4, HIGH);  // LED
    digitalWrite(5, HIGH);  // camera trigger
    delayMicroseconds(100);
    digitalWrite(5, LOW);  // camera trigger
    delayMicroseconds(delayVal-100+1);   // what happen if the result is 0?
    digitalWrite(4, LOW);  // LED
    // clear the string:
    inputString = "";
    stringComplete = false;
    delayVal = 0;
  }

}

void serialEvent() {
  while (Serial.available()) {
    // get the new byte:
    char inChar = (char)Serial.read();
    // add it to the inputString:
    inputString += inChar;
    // if the incoming character is a newline, set a flag so the main loop can
    // do something about it:
    if (inChar == '\n') {
      stringComplete = true;
    }
  }
}
