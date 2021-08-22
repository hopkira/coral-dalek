import sys
import paho.mqtt.client as mqtt

last_message = ""
client = mqtt.Client("dalek-python")
client.connect("localhost")

def on_message(client, userdata, message):
    """
    Enables the Dalek to receive a message from an Epruino Watch via
    MQTT over Bluetooth (BLE) to place it into active or inactive States
    """
    global last_message
    
    payload = str(message.payload.decode("utf-8"))
    if payload != last_message:
        last_message = payload
        event = payload[2:].lower()
        print("Event: ",str(event))
        #if command[1] == "Dale" and command[2] == "face" and command[3] == "on":
        #    dalek.on_event('waiting')
        #if command[1] == "Dale" and command[2] == "face" and command[3] == "off":
        #    dalek.on_event('silent')
        #else:
        #dalek.on_event('unknown')

client.on_message = on_message        # attach function to callback
client.subscribe("/ble/advertise/d3:fe:97:d2:d1:9e/espruino/m")

while True:
    try:
        client.loop(0.1)
    except KeyboardInterrupt:
        print("Stopped by CTRL+C")
        sys.exit(0)