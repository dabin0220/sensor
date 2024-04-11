import smbus
import RPi.GPIO as GPIO
import time

buzzer_pin = 21
GPIO.setmode(GPIO.BCM)
GPIO.setup(buzzer_pin, GPIO.OUT)
GPIO.output(buzzer_pin, GPIO.LOW)

bus = smbus.SMBus(1)

ADS_ADDR = 0x48
ADS_CONFIG_REG = 0x01
ADS_CONV_REG = 0x00

ADS_OS_SINGLE = 0x8000
ADS_MUX_0_1 = 0x4000    

def read_adc(channel):
    config = ADS_OS_SINGLE | ADS_MUX_0_1 | (channel << 12)
    bus.write_i2c_block_data(ADS_ADDR, ADS_CONFIG_REG, [(config >> 8) & 0xFF, config & 0xFF])
    time.sleep(0.1)
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
