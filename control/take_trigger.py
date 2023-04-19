"""
    Simple tool for trigger jai-camera via Arduino nano
"""

import serial
import time


class TakeTrigger:
    def __init__(self, port, trig_interval, led_duration):
        assert led_duration >= 0.1 and led_duration <= 16, led_duration
        self.trig_interval = trig_interval
        self.msg = f"{int(led_duration*10)}\n"
        self.read_timeout = 0.1
        self.s = serial.Serial(port, 115200)
        time.sleep(2)
        print(self.serial_read())

    def serial_read(self):
        t0 = time.time()
        data = b""
        while True:
            d = self.s.read(1)
            if len(d)>0:
                data += d
            if b"\n" in data:
                return data.decode()
            if time.time() - t0 > self.read_timeout:
                return None

    def run(self):
        while True:
            t0 = time.time()
            self.s.write(self.msg.encode())
            print(self.serial_read())
            dt = time.time()-t0
            assert dt < self.trig_interval, dt
            time.sleep(self.trig_interval - dt)

    def close(self):
        self.s.close()

    # context manager functions
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interval', '-i', help='Time interval between trigger (s).', type=float, default=1)
    parser.add_argument('--led-duration', '-l',
                        help='LED light duration (ms), values: 0.1 - 16 ms', type=float, default=1)
    parser.add_argument('--port', '-p', help='Serial port, default: /dev/ttyUSB0', default="/dev/ttyUSB0")
    args = parser.parse_args()

    trig_interval = args.interval
    led_duration = args.led_duration
    print(f"Set trigger interval: {trig_interval} s, LED light duration: {led_duration} ms")

    with TakeTrigger(args.port, trig_interval, led_duration) as take_trigger:
        take_trigger.run()
