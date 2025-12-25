"""
    Main cabbage detector and tracker
"""

from osgar.node import Node

class Cabbage(Node):
    def __init__(self, config, bus):
        super().__init__(config, bus)
        bus.register('detection')
        self.bus = bus
        self.verbose = False

    def on_image(self, data):
        pass

    def on_nmea_data(self, data):
        pass

