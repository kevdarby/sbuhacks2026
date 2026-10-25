"""Live guitar through the Mac's speakers/headphones. Nothing else.

No camera, no MIDI, no gesture tracking, no loop pedal - just the Katana's
USB input (ch1/2) boosted and played out whatever macOS's current default
output device is. Mute the amp's own master volume and practise quietly.

    uv run playback.py
    uv run playback.py --gain 26      # louder, for this run only
    uv run playback.py --save-gain    # persist --gain for future runs

Follows the default output device while running: connect AirPods (or
unplug headphones, or switch output in System Settings) and playback moves
across after a short gap. Ctrl-C to stop.
"""

import argparse
import ctypes
import ctypes.util
import time

import sounddevice as sd

from amp.audio import (
    DEVICE_NAME, MONITOR_GAIN_FILE, Looper, save_gain_db,
)

DEVICE_POLL_SECONDS = 0.75


# --- following the system default output device -------------------------
#
# PortAudio resolves device=None once, when the stream is opened, and it
# caches the whole device list at initialisation - so a stream keeps
# playing to the device it was opened on, and newly connected hardware is
# invisible until PortAudio is re-initialised. Neither is what you want
# when AirPods connect midway through practising.
#
# CoreAudio is asked directly for the current default output, which is
# cheap and doesn't disturb the running streams. Only its numeric ID is
# needed - any change means the default moved, which is the cue to rebuild
# the streams against the refreshed device list.

_KAUDIO_OBJECT_SYSTEM_OBJECT = 1


def _fourcc(code):
    return int.from_bytes(code.encode("ascii"), "big")


class _PropertyAddress(ctypes.Structure):
    _fields_ = [
        ("mSelector", ctypes.c_uint32),
        ("mScope", ctypes.c_uint32),
        ("mElement", ctypes.c_uint32),
    ]


def _load_coreaudio():
    path = ctypes.util.find_library("CoreAudio")
    return ctypes.CDLL(path) if path else None


_COREAUDIO = _load_coreaudio()


def default_output_id():
    """CoreAudio's current default output device ID, or None if it can't be
    read (in which case device-change following is simply skipped)."""
    if _COREAUDIO is None:
        return None
    addr = _PropertyAddress(
        _fourcc("dOut"),  # kAudioHardwarePropertyDefaultOutputDevice
        _fourcc("glob"),  # kAudioObjectPropertyScopeGlobal
        0,                # kAudioObjectPropertyElementMain
    )
    device_id = ctypes.c_uint32(0)
    size = ctypes.c_uint32(ctypes.sizeof(device_id))
    err = _COREAUDIO.AudioObjectGetPropertyData(
        ctypes.c_uint32(_KAUDIO_OBJECT_SYSTEM_OBJECT), ctypes.byref(addr),
        0, None, ctypes.byref(size), ctypes.byref(device_id),
    )
    return None if err != 0 else device_id.value


def refresh_portaudio():
    """Rebuild PortAudio's cached device list so a device connected after
    startup can actually be opened. Every existing stream must already be
    closed - this invalidates them all."""
    try:
        sd._terminate()
        sd._initialize()
    except Exception:
        pass  # worst case we keep the old list and stay on the old device


def output_name():
    try:
        return sd.query_devices(sd.default.device[1])["name"]
    except Exception:
        return "(unknown)"


# --- playback -----------------------------------------------------------

def play_until_device_changes(looper):
    """Opens the Katana input and the Mac output, and streams between them
    until the default output device changes or the user interrupts.
    Returns True if it should reconnect to a new device."""
    audio_stream = looper.open_stream()
    mac_stream = looper.open_mac_stream()
    started_on = default_output_id()

    with audio_stream, mac_stream:
        print(f"  -> playing through: {output_name()}")
        if not mac_stream.active:
            print("     WARNING: output stream did not start.")
        while True:
            time.sleep(DEVICE_POLL_SECONDS)
            if looper.cb_errors or looper.mac_cb_errors:
                print(f"     audio error: {looper.last_error}")
                looper.cb_errors = looper.mac_cb_errors = 0
            now = default_output_id()
            if started_on is not None and now is not None and now != started_on:
                print("  -- default output changed, reconnecting...")
                return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gain", type=float, default=None, metavar="DB",
        help="Monitor gain in dB. Defaults to the saved value (the same one "
             "main.py's +/- keys tune). The guitar signal arrives at "
             "pickup level, so this needs real boost - around +20dB.",
    )
    parser.add_argument(
        "--save-gain", action="store_true",
        help="Persist --gain so main.py and future runs use it too.",
    )
    args = parser.parse_args()

    looper = Looper()
    if args.gain is not None:
        looper.monitor_gain_db = args.gain
        if args.save_gain:
            save_gain_db(MONITOR_GAIN_FILE, args.gain)
            print(f"Saved monitor gain {args.gain:+.1f}dB.")

    # The live monitor is the entire point here, so it is simply on - there
    # is no toggle to forget, and no window that has to have keyboard focus
    # for one to work.
    looper.monitor_enabled = True

    print(f"Live guitar -> Mac. Gain {looper.monitor_gain_db:+.1f}dB. "
          f"Ctrl-C to stop.")

    try:
        while True:
            try:
                if not play_until_device_changes(looper):
                    break
            except sd.PortAudioError as exc:
                print(f"  !! could not open audio: {exc}")
                if DEVICE_NAME.lower() not in str(
                        sd.query_devices()).lower():
                    print(f"     No '{DEVICE_NAME}' device found - is the amp "
                          f"connected and powered on?")
                    break
                print("     retrying in 2s...")
                time.sleep(2.0)
            # Rebuild the device list before reopening, so a device that was
            # connected after startup is visible and device=None resolves to
            # it. Safe here: both streams are closed by now.
            refresh_portaudio()
            looper.reset_monitor()  # the buffered audio is stale by now
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
