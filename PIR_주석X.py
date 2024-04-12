import RPi.GPIO as GPIO
import time

PIR_PIN = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIR_PIN, GPIO.IN)

person_count = 0

try:
    print("PIR 센서 동작 시작")
    while True:
        if GPIO.input(PIR_PIN): 
            nowtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print("현재시간 : {}".format(nowtime))
            print("사람이 감지되었습니다.")
            person_count += 1  
            print("현재 감지한 사람 수:", person_count)
            time.sleep(5)
        else:
            print("감지되지않음") 
            time.sleep(3)
            
except KeyboardInterrupt:
    print("키보드 인터럽트가 감지되었습니다. 정리 중...")
    GPIO.cleanup()
