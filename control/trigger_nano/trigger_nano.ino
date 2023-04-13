String inputString = "";
bool stringComplete = false;

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
    Serial.println(inputString);
    digitalWrite(4, HIGH);  // LED
    digitalWrite(5, HIGH);  // camera trigger
    delay(1);
    digitalWrite(4, LOW);  // LED
    digitalWrite(5, LOW);  // camera trigger
    // clear the string:
    inputString = "";
    stringComplete = false;
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
