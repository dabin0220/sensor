import smbus          # import SMBus module of I2C
from time import sleep # import sleep for delay
import math           # 수학 연산을 위한 math 모듈 임포트
import RPi.GPIO as GPIO

GPIO.setwarnings(False)

# MPU6050 레지스터와 그 주소 정의
PWR_MGMT_1   = 0x6B
SMPLRT_DIV   = 0x19
CONFIG       = 0x1A
GYRO_CONFIG  = 0x1B
INT_ENABLE   = 0x38
ACCEL_XOUT_H = 0x3B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F
GYRO_XOUT_H  = 0x43
GYRO_YOUT_H  = 0x45
GYRO_ZOUT_H  = 0x47

LED_PIN = 23
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)

bus = smbus.SMBus(1)   # I2C 버스 1 사용 (구형 보드는 smbus.SMBus(0) 사용)
Device_Address = 0x68  # MPU6050 장치 주소

def MPU_Init():
    # 샘플 레이트 레지스터 설정
    bus.write_byte_data(Device_Address, SMPLRT_DIV, 7)
    
    # 전원 관리 레지스터 설정
    bus.write_byte_data(Device_Address, PWR_MGMT_1, 1)
    
    # 설정 레지스터 설정
    bus.write_byte_data(Device_Address, CONFIG, 0)
    
    # 자이로 설정 레지스터 설정
    bus.write_byte_data(Device_Address, GYRO_CONFIG, 24)
    
    # 인터럽트 활성화 레지스터 설정
    bus.write_byte_data(Device_Address, INT_ENABLE, 1)

def read_raw_data(addr):
    # 가속도 및 자이로 데이터는 16비트
    high = bus.read_byte_data(Device_Address, addr)
    low = bus.read_byte_data(Device_Address, addr + 1)
    
    # 상위 및 하위 바이트 결합
    value = ((high << 8) | low)
    
    # MPU6050에서 부호 있는 값을 얻기 위한 처리
    if value > 32768:
        value = value - 65536
    return value

def get_accel_data():
    acc_x = read_raw_data(ACCEL_XOUT_H)
    acc_y = read_raw_data(ACCEL_YOUT_H)
    acc_z = read_raw_data(ACCEL_ZOUT_H)
    
    Ax = acc_x / 16384.0
    Ay = acc_y / 16384.0
    Az = acc_z / 16384.0
    
    return Ax, Ay, Az

# 이동 평균 필터 적용 함수
def apply_moving_average(data, window_size=5):
    if len(data) < window_size:
        return sum(data) / len(data)
    return sum(data[-window_size:]) / window_size

accData = [0, 0, 0]
window_size = 10  # 이동 평균 필터 창 크기

# 초기화
try:
    MPU_Init()
    x_data, y_data, z_data = [], [], []
    for i in range(10):
        Ax, Ay, Az = get_accel_data()
        x_data.append(Ax)
        y_data.append(Ay)
        z_data.append(Az)
        accData[0] += Ax
        accData[1] += Ay
        accData[2] += Az
        sleep(0.1)
    
    accData[0] /= 10
    accData[1] /= 10
    accData[2] /= 10
    
    print("Initial Acceleration Data:", accData)
    
    threshold = 0.2  # 충격 감지 임계값(g 단위) - 더 작은 값으로 설정
    duration = 1  # 지속 시간 (초)
    check_interval = 0.01  # 데이터 샘플링 간격 (초) - 더 짧은 간격으로 설정
    required_samples = int(duration / check_interval)  # 지속 시간 동안 필요한 샘플 수
    
    while True:
        shock_detected = False
        for _ in range(required_samples):
            Ax, Ay, Az = get_accel_data()
            x_data.append(Ax)
            y_data.append(Ay)
            z_data.append(Az)
            
            # 이동 평균 필터 적용
            Ax = apply_moving_average(x_data, window_size)
            Ay = apply_moving_average(y_data, window_size)
            Az = apply_moving_average(z_data, window_size)
            
            # 벡터 크기 계산
            acceleration_magnitude = math.sqrt((Ax - accData[0])**2 + (Ay - accData[1])**2 + (Az - accData[2])**2)
            
            if acceleration_magnitude >= threshold:
                shock_detected = True
                break  # 충격이 감지되면 즉시 루프를 종료하여 다음 지점부터 새로운 충격을 찾음
            sleep(check_interval)
        
        if shock_detected:
            print("<<<<< 충격 발생 >>>>>")
            GPIO.output(LED_PIN, GPIO.HIGH)
        else:
            GPIO.output(LED_PIN, GPIO.LOW)
        sleep(0.1)  # 작은 지연 추가

except KeyboardInterrupt:
    pass
