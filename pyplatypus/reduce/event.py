# -*- coding: utf-8 -*-
"""
unpack streaming file
@author: andrew
"""
import numpy as np

def events(f, endoflastevent=127):
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
        
    Returns
    -------
    (x_events, y_events, t_events, f_events), endoflastevent:
        x_events, y_events, t_events and f_events are numpy arrays containing
        the events. endoflastevent is a byte offset to the end of the last
        successful event read from the file. Use this value to extract more
        events from the same file at a future date.
    """
    if not f:
        return None

    state = 0L
    event_ended = 0L
    frame_number = -1L
    dt = 0L
    t = 0L
    x = -0L
    y = -0L

    x_events = np.array((), dtype='int32')
    y_events = np.array((), dtype='int32')
    t_events = np.array((), dtype='uint32')
    f_events = np.array((), dtype='int32')

    BUFSIZE = 16384

    while True:
        x_neutrons = []
        y_neutrons = []
        t_neutrons = []
        f_neutrons = []

        f.seek(endoflastevent + 1)
        buf = f.read(BUFSIZE)

        filepos = endoflastevent + 1

        if not len(buf):
            break

        buf = map(ord, buf)
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

    return (x_events, y_events, t_events, f_events), endoflastevent