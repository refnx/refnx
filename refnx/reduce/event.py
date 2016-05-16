# -*- coding: utf-8 -*-
"""
Unpack event files
"""
import numpy as np
HAVE_CEVENTS = False
try:
    from refnx.reduce._cevent import _cevents
    HAVE_CEVENTS = True
except ImportError:
    pass


def process_event_stream(events, frame_bins, t_bins, y_bins, x_bins):
    """
    Processes the event mode dataset into a histogram.

    Parameters
    ----------
    events : tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        the 4-tuple of events (F, T, Y, X)
    frame_bins : array_like
        specifies the frame bins required in the image.
    t_bins : array_like
        specifies the time bins required in the image
    y_bins : array_like
        specifies the y bins required in the image
    x_bins : array_like
        specifies the x bins required in the image

    Returns
    -------
    detector, frame_bins : np.ndarray, np.ndarray
        The new detector image and the amended frame bins.

    Notes
    -----
    The frame bins that are supplied to this function are truncated to 0 and
    the maximum frame bin in the events. Thus, if
    frame_bins = [-20, -10, 5, 10, 20, 30] and the maximum frame bin is 25 then
    the amended frame_bins is calculated as [0, 5, 10, 20, 25], i.e. 4 bins are
    returned instead of 5.
    """
    max_frame = max(events[0])
    frame_bins = np.sort(frame_bins)

    # truncate the limits of frame bins
    frame_bins = np.unique(np.clip(frame_bins, 0, max_frame))

    localxbins = np.array(x_bins)
    localybins = np.array(y_bins)
    localtbins = np.sort(np.array(t_bins))
    localframe_bins = np.array(frame_bins)
    reversed_x, reversed_y = False, False

    if localxbins[0] > localxbins[-1]:
        localxbins = localxbins[::-1]
        reversed_x = True

    if localybins[0] > localybins[-1]:
        localybins = localybins[::-1]
        reversed_y = True

    detector, edge = np.histogramdd(events, bins=(localframe_bins,
                                                  localtbins,
                                                  localybins,
                                                  localxbins))
    if reversed_x:
        detector = detector[:, :, :, ::-1]

    if reversed_y:
        detector = detector[:, :, ::-1, :]

    return detector, localframe_bins


def events(f, end_last_event=127, max_frames=np.inf):
    """
    Unpacks event data from packedbinary format for the ANSTO Platypus
    instrument

    Parameters
    ----------

    f : file-like or str
        The file to read the data from. If `f` is not file-like then f is
        assumed to be a path pointing to the event file.
    end_last_event : uint
        The reading of event data starts from `end_last_event + 1`. The default
        of 127 corresponds to a file header that is 128 bytes long.
    max_frames : int
        Stop reading the event file when have read this many frames.

    Returns
    -------
    (f_events, t_events, y_events, x_events), end_last_event:
        x_events, y_events, t_events and f_events are numpy arrays containing
        the events. end_last_event is a byte offset to the end of the last
        successful event read from the file. Use this value to extract more
        events from the same file at a future date.
    """
    if HAVE_CEVENTS:
        return _cevents(f, end_last_event=end_last_event,
                        max_frames=max_frames)
    else:
        return _events(f, end_last_event=end_last_event,
                        max_frames=max_frames)


def _events(f, end_last_event=127, max_frames=np.inf):
    """
    Unpacks event data from packedbinary format for the ANSTO Platypus
    instrument
    
    Parameters
    ----------
    
    f : file-like or str
        The file to read the data from. If `f` is not file-like then f is
        assumed to be a path pointing to the event file.
    end_last_event : uint
        The reading of event data starts from `end_last_event + 1`. The default
        of 127 corresponds to a file header that is 128 bytes long.
    max_frames : int
        Stop reading the event file when have read this many frames.
        
    Returns
    -------
    (f_events, t_events, y_events, x_events), end_last_event:
        x_events, y_events, t_events and f_events are numpy arrays containing
        the events. end_last_event is a byte offset to the end of the last
        successful event read from the file. Use this value to extract more
        events from the same file at a future date.
    """
    fi = f
    auto_f = None
    if not hasattr(fi, 'read'):
        auto_f = open(f, 'rb')
        fi = auto_f

    frame_number = -1
    dt = 0
    t = 0
    x = -0
    y = -0

    x_events = np.array((), dtype='int32')
    y_events = np.array((), dtype='int32')
    t_events = np.array((), dtype='uint32')
    f_events = np.array((), dtype='int32')

    bufsize = 32768

    while True and frame_number < max_frames:
        x_neutrons = []
        y_neutrons = []
        t_neutrons = []
        f_neutrons = []

        fi.seek(end_last_event + 1)
        buf = fi.read(bufsize)

        filepos = end_last_event + 1

        if not len(buf):
            break

        buf = bytearray(buf)
        state = 0

        for i, c in enumerate(buf):
            if state == 0:
                x = c
                state += 1
            elif state == 1:
                x |= (c & 0x3) * 256

                if x & 0x200:
                    x = - (0x100000000 - (x | 0xFFFFFC00))
                y = int(c / 4)
                state += 1
            else:
                if state == 2:
                    y |= (c & 0xF) * 64

                    if y & 0x200:
                        y = -(0x100000000 - (y | 0xFFFFFC00))
                event_ended = ((c & 0xC0) != 0xC0 or state >= 7)

                if not event_ended:
                    c &= 0x3F
                if state == 2:
                    dt = c >> 4
                else:
                    dt |= c << 2 + 6 * (state - 3)

                if not event_ended:
                    state += 1
                else:
                    # print "got to state", state, event_ended, x, y, frame_number, t, dt
                    state = 0
                    end_last_event = filepos + i
                    if x == 0 and y == 0 and dt == 0xFFFFFFFF:
                        t = 0
                        frame_number += 1
                        if frame_number == max_frames:
                            break
                    else:
                        t += dt
                        if frame_number == -1:
                            return None
                        x_neutrons.append(x)
                        y_neutrons.append(y)
                        t_neutrons.append(t)
                        f_neutrons.append(frame_number)

        if len(x_neutrons):
            x_events = np.append(x_events, x_neutrons)
            y_events = np.append(y_events, y_neutrons)
            t_events = np.append(t_events, t_neutrons)
            f_events = np.append(f_events, f_neutrons)
    
    t_events //= 1000

    if auto_f:
        auto_f.close()

    return (f_events, t_events, y_events, x_events), end_last_event
