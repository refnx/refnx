# -*- coding: utf-8 -*-
"""
unpack streaming file
@author: andrew
"""
import numpy as np


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
    x_bins : array_like
        specifies the x bins required in the image
    y_bins : array_like
        specifies the y bins required in the image

    Returns
    -------
    detector, frame_bins : np.ndarray, np.ndarray
        The new detector image and the amended frame bins. The frame bins that
        are supplied to this function are truncated to 0 and the maximum frame
        bin in the event file.
    """
    max_frame = max(events[0])
    frame_bins = np.sort(frame_bins)

    # truncate the lower limit of frame bins to be 0 if they exceed it.
    loc = np.searchsorted(frame_bins, 0)
    frame_bins = frame_bins[loc:]
    frame_bins[0] = 0

    # truncate the upper limit of frame bins to be the max frame number
    # if they exceed it.
    # loc = np.searchsorted(frame_bins, max_frame)
    # frame_bins = frame_bins[: loc]

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


def events(f, endoflastevent=127, max_frames=np.inf):
    """
    Unpacks event data from packedbinary format for the ANSTO Platypus
    instrument
    
    Parameters
    ----------
    
    f : file
        The file to read the data from.
    endoflastevent : uint
        The file position to start the read from. The data starts from byte
        127.
    max_frames : int
        Stop reading the event file when you get to this many frames.
        
    Returns
    -------
    (f_events, t_events, y_events, x_events), endoflastevent:
        x_events, y_events, t_events and f_events are numpy arrays containing
        the events. endoflastevent is a byte offset to the end of the last
        successful event read from the file. Use this value to extract more
        events from the same file at a future date.
    """
    if not f:
        return None

    state = 0
    event_ended = 0
    frame_number = -1
    dt = 0
    t = 0
    x = -0
    y = -0

    x_events = np.array((), dtype='int32')
    y_events = np.array((), dtype='int32')
    t_events = np.array((), dtype='uint32')
    f_events = np.array((), dtype='int32')

    BUFSIZE = 32768

    while True and frame_number < max_frames:
        x_neutrons = []
        y_neutrons = []
        t_neutrons = []
        f_neutrons = []

        f.seek(endoflastevent + 1)
        buf = f.read(BUFSIZE)

        filepos = endoflastevent + 1

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
                    y = y | ((c & 0xF) * 64)

                    if y & 0x200:
                        y = -(0x100000000 - (y | 0xFFFFFC00))
                event_ended = ((c & 0xC0) != 0xC0 or state >= 7)

                if not event_ended:
                    c &= 0x3F
                if state == 2:
                    dt = c >> 4
                else:
                    dt |= (c) << (2 + 6 * (state - 3))

                if not event_ended:
                    state += 1
                else:
                    #print "got to state", state, event_ended, x, y, frame_number, t, dt
                    state = 0
                    endoflastevent = filepos + i
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
    
    t_events = t_events // 1000
    return (f_events, t_events, y_events, x_events), endoflastevent