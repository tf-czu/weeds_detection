"""
    Generator of coordinates in NMEA format.
"""

import math
import datetime
from threading import Thread


class NMEAGenerator:
    def __init__(self, config, bus):
        self.bus = bus
        self.bus.register("raw")
        self.input_thread = Thread(target=self.run_input, daemon=True)

        self.lat = config.get("start_lat", 50.0755)
        self.lon = config.get("start_lon", 14.4378)
        self.speed = config.get("speed_mps", 0.3)
        self.heading = math.radians(config.get("heading_deg", 90))
        self.interval = 1.0 / config.get("frequency_hz", 5)

        # Simulation starts now
        self.current_time = datetime.datetime.utcnow()
        self.R_EARTH = 6378137.0  # Earth radius (m)

    def start(self):
        self.input_thread.start()

    def join(self, timeout=None):
        self.input_thread.join(timeout=timeout)


    def _decimal_to_nmea_coord(self, decimal_degrees, is_latitude):
        """
        Convert regular coordinates ti (50.128) to NMEA formát (5007.68,N)
        """
        degrees = int(abs(decimal_degrees))
        minutes = (abs(decimal_degrees) - degrees) * 60.0

        # Formating:
        # Latitude: DDMM.MMMMMM
        # Longitude: DDDMM.MMMMMM (3 digits for degrees)
        if is_latitude:
            coord_str = f"{degrees:02d}{minutes:011.8f}"
            hemisphere = 'N' if decimal_degrees >= 0 else 'S'
        else:
            coord_str = f"{degrees:03d}{minutes:011.8f}"
            hemisphere = 'E' if decimal_degrees >= 0 else 'W'

        return coord_str, hemisphere

    def _calculate_checksum(self, sentence_body):
        calc_cksum = 0
        for s in sentence_body:
            calc_cksum ^= ord(s)
        return f"{calc_cksum:02X}"

    def _move_position(self):
        """
        Calculate new coordinates based on speed and directions.
        """
        distance = self.speed * self.interval
        dy = distance * math.cos(self.heading)
        dx = distance * math.sin(self.heading)

        # dLat = dy / R
        delta_lat = (dy / self.R_EARTH) * (180 / math.pi)

        # dLon = dx / (R * cos(lat))
        delta_lon = (dx / (self.R_EARTH * math.cos(math.radians(self.lat)))) * (180 / math.pi)

        self.lat += delta_lat
        self.lon += delta_lon

        # Time
        self.current_time += datetime.timedelta(seconds=self.interval)

    def generate_gngga(self):
        """
        Generate one GNGGA message
        """
        self._move_position()
        time_str = self.current_time.strftime("%H%M%S.%f")[:9]
        lat_str, lat_dir = self._decimal_to_nmea_coord(self.lat, is_latitude=True)
        lon_str, lon_dir = self._decimal_to_nmea_coord(self.lon, is_latitude=False)

        quality = "5"
        num_sats = "09"
        hdop = "1.9"
        altitude = "290.1985"
        geoid_sep = "45.0552"
        diff_age = "01"  # Age of differential correction
        station_id = "0533"

        # Make the body
        # Sample msg: GNGGA,190615.40,5007.70786799,N,01422.49430110,E,2,09,1.9,290.1985,M,45.0552,M,01,0533
        body = (f"GNGGA,{time_str},{lat_str},{lat_dir},{lon_str},{lon_dir},"
                f"{quality},{num_sats},{hdop},{altitude},M,{geoid_sep},M,"
                f"{diff_age},{station_id}")

        # Add checksum
        checksum = self._calculate_checksum(body)
        nmea_message = f"${body}*{checksum}"

        # Return msq like b'...'
        return nmea_message.encode('ascii')

    def run_input(self):
        while self.bus.is_alive():
            msg = self.generate_gngga()
            self.bus.publish("raw", msg)
            self.bus.sleep(self.interval)

    def request_stop(self):
        self.bus.shutdown()
