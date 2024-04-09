import smbus
import RPi.GPIO as GPIO
import time

buzzer_pin = 21
GPIO.setmode(GPIO.BCM)
GPIO.setup(buzzer_pin, GPIO.OUT)
GPIO.output(buzzer_pin, GPIO.LOW)

# I2C 인터페이스 설정
bus = smbus.SMBus(1)  # 라즈베리 파이 3 이상일 경우 1, 이하일 경우 0 사용

# ADS1115 주소 및 설정
ADS_ADDR = 0x48  # ADS1115의 기본 주소는 0x48입니다.
ADS_CONFIG_REG = 0x01  # 설정 레지스터 주소
ADS_CONV_REG = 0x00    # 변환 레지스터 주소

# 설정 값
ADS_OS_SINGLE = 0x8000  # 단일 변환 모드
ADS_MUX_0_1 = 0x4000     # AIN0과 AIN1을 사용하여 단일 채널 측정

# I2C 통신을 통해 ADC 값을 읽는 함수
def read_adc(channel):
    config = ADS_OS_SINGLE | ADS_MUX_0_1 | (channel << 12)
    bus.write_i2c_block_data(ADS_ADDR, ADS_CONFIG_REG, [(config >> 8) & 0xFF, config & 0xFF])
    time.sleep(0.1)  # 변환이 완료되기를 기다립니다.
    data = bus.read_i2c_block_data(ADS_ADDR, ADS_CONV_REG, 2)
    return (data[0] << 8 | data[1])

try:
    while True:
        nowtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        gas_value = read_adc(0)
        print("현재시간 : {}".format(nowtime))
        print("Gas Sensor Value:", gas_value)

        if gas_value >= 5000:
            GPIO.output(buzzer_pin, GPIO.HIGH)
            time.sleep(0.5)
            GPIO.output(buzzer_pin, GPIO.LOW)
            time.sleep(0.5)
        else:
            GPIO.output(buzzer_pin, GPIO.LOW)
        
        time.sleep(1)

except KeyboardInterrupt:
    pass

GPIO.cleanup()
