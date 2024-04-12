import RPi.GPIO as GPIO
import time
import Adafruit_DHT as dht

DHT_PIN = 4
LED_PIN = 24

GPIO.setmode(GPIO.BCM)
GPIO.setup(DHT_PIN, GPIO.IN)
GPIO.setup(LED_PIN, GPIO.OUT)

while True:
    try:
        nowtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        humidity, temperature_c = dht.read_retry(dht.DHT22, DHT_PIN)
        if temperature_c is not None:
            temperature_f = temperature_c * (9 / 5) + 32  
            print("현재시간 : {}".format(nowtime))
            print(
                "온도: {:.1f} F / {:.1f} C  습도: {:.1f}% ".format(temperature_f, temperature_c, humidity)
            )   
            
            if temperature_c >= 25:
                print("<<<에어컨 작동 요망>>>")
                GPIO.output(LED_PIN, GPIO.HIGH)
            elif temperature_c <= 17:
                print("<<<난방기 작동 요망>>>")
                GPIO.output(LED_PIN, GPIO.HIGH)
            else:
                GPIO.output(LED_PIN, GPIO.LOW)
        else:
            print("온도 데이터 실패")
    
    except RuntimeError as error:
        print(error.args[0])
        time.sleep(2.0)
        continue
    except Exception as error:
        raise error
    
    time.sleep(2.0)
