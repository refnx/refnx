from __future__ import division, absolute_import
import numpy as np

cimport numpy as np
cimport cython
from cython.view cimport array as cvarray

ii32 = np.iinfo(np.int32)


@cython.cdivision(False)
def _cevents(f,
             int end_last_event=127,
             max_frames=None):
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
    max_frames : None, int
        Stop reading the event file when have read this many frames.

    Returns
    -------
    (f_events, t_events, y_events, x_events), end_last_event:
        x_events, y_events, t_events and f_events are numpy arrays containing
        the events. end_last_event is a byte offset to the end of the last
        successful event read from the file. Use this value to extract more
        events from the same file at a future date.
    """
    if max_frames is None:
        max_frames = ii32.max
    max_frames = int(max_frames)

    fi = f
    auto_f = None
    if not hasattr(fi, 'read'):
        auto_f = open(f, 'rb')
        fi = auto_f

    cdef int frame_number = -1
    cdef unsigned int dt = 0
    cdef unsigned int t = 0
    cdef int x = -0
    cdef int y = -0
    cdef int state = 0
    cdef int i = 0
    cdef unsigned char c
    cdef int event_ended = 0
    cdef int num_events = 0
    cdef int filepos = 0

    cdef np.ndarray[np.int32_t, ndim=1] x_events = np.array((), dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] y_events = np.array((), dtype=np.int32)
    cdef np.ndarray[np.uint32_t, ndim=1] t_events = np.array((), dtype=np.uint32)
    cdef np.ndarray[np.int32_t, ndim=1] f_events = np.array((), dtype=np.int32)

    cdef int bufsize = 32768

    # these are buffers to store events from each read of the file
    # use of buffers prevents continual allocation of memory.
    cdef np.ndarray[np.int32_t, ndim=1] x_neutrons = np.zeros((bufsize), dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] y_neutrons = np.zeros((bufsize), dtype=np.int32)
    cdef np.ndarray[np.uint32_t, ndim=1] t_neutrons = np.zeros((bufsize), dtype=np.uint32)
    cdef np.ndarray[np.int32_t, ndim=1] f_neutrons = np.zeros((bufsize), dtype=np.int32)

    while True and frame_number < max_frames:
        num_events = 0
        # TODO: possibly re-szeo *_neutrons?

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
                        x_neutrons[num_events] = x
                        y_neutrons[num_events] = y
                        t_neutrons[num_events] = t
                        f_neutrons[num_events] = frame_number
                        num_events += 1

        if len(x_neutrons):
            x_events = np.append(x_events, x_neutrons[0:num_events])
            y_events = np.append(y_events, y_neutrons[0:num_events])
            t_events = np.append(t_events, t_neutrons[0:num_events])
            f_events = np.append(f_events, f_neutrons[0:num_events])

    t_events //= 1000

    if auto_f:
        auto_f.close()

    return (f_events, t_events, y_events, x_events), end_last_event
