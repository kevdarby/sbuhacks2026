"""The USB relay that switches 12V to the drum solenoid. It's a serial
device (JESSINIE/LCUS-style, 9600 baud), so the Mac drives it directly -
no microcontroller in between.
"""

import glob
import time

import serial

# Which relay the solenoid is wired to. The first board's channel 1 stopped
# passing current - most likely its contacts burned, from switching the
# solenoid with no flyback diode across it (a 1N4007/1N5819 across the
# solenoid, stripe on the +12V side, absorbs the spike instead). Now on
# channel 4 of a 4-channel board.
DEFAULT_CHANNEL = 4


def relay_command(channel, on):
    """A0 <channel> <state> <checksum>, the checksum being the byte sum."""
    body = [0xA0, channel, 1 if on else 0]
    return bytes(body + [sum(body) & 0xFF])


RELAY_ON = relay_command(1, True)
RELAY_OFF = relay_command(1, False)

# Longer than this doesn't hit harder, it just heats the coil
MAX_PULSE = 0.1
# The relay needs about this long open between hits to keep up
RELAY_RECOVERY = 0.05


def find_port():
    ports = glob.glob("/dev/cu.usbserial*")
    if not ports:
        raise SystemExit("No USB relay found (no /dev/cu.usbserial*) - is it plugged in?")
    return ports[0]


class Relay:
    def __init__(self, port=None, channel=None):
        self.port = port or find_port()
        self.channel = channel or DEFAULT_CHANNEL
        self._on = relay_command(self.channel, True)
        self._off = relay_command(self.channel, False)
        self._serial = serial.Serial(self.port, 9600, timeout=1)
        time.sleep(0.5)  # let the board settle after the port opens

    def hit(self, pulse):
        """Close the relay for `pulse` seconds. Blocks for that long."""
        self._serial.write(self._on)
        self._serial.flush()
        time.sleep(min(pulse, MAX_PULSE))
        self._serial.write(self._off)
        self._serial.flush()

    def close(self):
        # Never leave the coil energized, however we exit
        self._serial.write(self._off)
        self._serial.flush()
        self._serial.close()


class SilentRelay:
    """Stands in for Relay when the drums should stay quiet but everything
    driven by the beat (e.g. the bass) should still run. Takes the same
    time per hit, so timing behaves exactly as with the real relay."""

    port = "silent"

    def hit(self, pulse):
        time.sleep(min(pulse, MAX_PULSE))

    def close(self):
        pass
