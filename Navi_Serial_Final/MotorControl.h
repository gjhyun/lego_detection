class MotorControl{
  public:
    byte IN_pin;
    byte PWM_pin;
 
    int PWM_val;

  public: 
    MotorControl(byte IN_pin, byte PWM_pin){
      this->IN_pin = IN_pin;
      this->PWM_pin = PWM_pin;
      pinMode(this->IN_pin, OUTPUT);
      pinMode(this->PWM_pin, OUTPUT);
    }

   void setPWM(int PWM_val){
      this->PWM_val = PWM_val;
      analogWrite(this->PWM_pin, this->PWM_val);
   }
   
   void forward(){
      digitalWrite(this->IN_pin, HIGH);
   }

   void backward(){
      digitalWrite(this->IN_pin, LOW);
   }

   void halt() {
      analogWrite(this->PWM_pin, 0);
      digitalWrite(this->IN_pin, LOW);
   }
};
