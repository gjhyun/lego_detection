#!/usr/bin/env python3
# Allows Alexa to command the Balboa to stand up or fall down
# Derived from examples in the Flask-Ask repo: https://github.com/johnwheeler/flask-ask

import serial

from flask import Flask
from flask_ask import Ask, statement
from random import randrange

app = Flask(__name__)
ask = Ask(app, '/')

@ask.intent('startMoving')
def start_moving():
    speech_text = 'I/m moving'
    ser.write(b'm')
    return statement(speech_text).simple_card('Wilson', speech_text)

@ask.intent('stopMoving')
def stop_moving():
    speech_text = "I/m stopping"
    ser.write(b's')
    return statement(speech_text).simple_card('Wilson', speech_text)

@ask.intent('makeMeLaugh')
def make_me_laugh():
    a = randrange(7)
    if a == 0:
        speech_text = 'I can’t believe I got fired from the calendar factory: all I did was take a day off!'
    elif a == 1:
        speech_text = 'Most people are shocked when they find out how bad I am as an electrician.'
    elif a == 2:
        speech_text = 'After suffering weak gain at the poles, the National Transistor Party has been trying to energize their base.'
    elif a == 3:
        speech_text = 'Looking for a boyfriend in engineering: the odds are good, but the goods are odd.'
    elif a == 4:
        speech_text = 'You wanted a joke? Just look at your GPA'
    elif a == 5:
        speech_text = 'Welcome to college, where every single person is smarter than you except for the 3 people in your group project.'
    else:
        speech_text = 'Power naps are great. You can build up charge with them.'

    return statement(speech_text).simple_card('Wilson',speech_text)

if __name__ == '__main__':
    ser = serial.Serial('/dev/ttyACM0', 9600)
    app.run()

