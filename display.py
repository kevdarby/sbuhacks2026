"""The shape of the screen, so the camera picture can be cropped to fill it.

OpenCV shows a picture at its own aspect ratio, so in fullscreen a 16:9
camera on a 16:10 MacBook screen left a bar at the top. Cropping each frame
to the screen's shape (below the notch, on MacBooks that have one - the
strip beside the notch stays black in fullscreen) makes it fill exactly.
"""

import ctypes
import ctypes.util

# If the screen can't be asked: a 16:10 MacBook screen less a notch strip
FALLBACK_ASPECT = 1470 / 924


class _Size(ctypes.Structure):
    _fields_ = [("width", ctypes.c_double), ("height", ctypes.c_double)]


class _Rect(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double), ("size", _Size)]


class _Insets(ctypes.Structure):
    _fields_ = [("top", ctypes.c_double), ("left", ctypes.c_double),
                ("bottom", ctypes.c_double), ("right", ctypes.c_double)]


def screen_aspect():
    """Width / height of the main screen's usable area in fullscreen: its
    frame less the notch's safe-area inset. Asks AppKit through the
    Objective-C runtime (no PyObjC needed); falls back on failure."""
    try:
        ctypes.CDLL("/System/Library/Frameworks/AppKit.framework/AppKit")
        objc = ctypes.CDLL(ctypes.util.find_library("objc"))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.restype = ctypes.c_void_p

        def send(obj, sel, restype, *args):
            fn = ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p)(
                ("objc_msgSend", objc))
            return fn(obj, objc.sel_registerName(sel.encode()), *args)

        screen = send(objc.objc_getClass(b"NSScreen"), "mainScreen", ctypes.c_void_p)
        if not screen:
            return FALLBACK_ASPECT
        frame = send(screen, "frame", _Rect)
        top = 0.0
        try:
            top = send(screen, "safeAreaInsets", _Insets).top  # macOS 12+
        except Exception:
            pass
        height = frame.size.height - top
        if frame.size.width <= 0 or height <= 0:
            return FALLBACK_ASPECT
        return frame.size.width / height
    except Exception:
        return FALLBACK_ASPECT


def crop_to_aspect(frame, aspect):
    """The centre of `frame` at `aspect` (width / height), as a view."""
    h, w = frame.shape[:2]
    if w / h > aspect:
        cw = int(round(h * aspect))
        x = (w - cw) // 2
        return frame[:, x:x + cw]
    ch = int(round(w / aspect))
    y = (h - ch) // 2
    return frame[y:y + ch]
