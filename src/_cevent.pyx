# cython: language_level=3, cdivision=False
import numpy as np

cimport numpy as cnp
cimport cython

from libc.stdio cimport fopen, fclose, FILE, EOF, fseek, SEEK_END, SEEK_SET
from libc.stdio cimport ftell, fgetc, fgets, getc, gets, feof, fread, getline

ii32 = np.iinfo(np.int32)

"""
Notes
-----
Example extract of file:

state, c, filepos
7 63 9363420
6 28 9363427
6 12 9363434
7 63 9363442
6 18 9363449
7 63 9363457
6 9 9363464
7 63 9363472
6 11 9363479
5 35 9363485
6 15 9363492
7 63 9363500
6 9 9363507
6 14 9363514
7 63 9363522
6 21 9363529
7 63 9363537
7 63 9363545
6 25 9363552
7 63 9363560
6 13 9363567
5 17 9363573

- state will always be 7 when c==63, this is always the end of an event
- events can also end if state isn't 7.
- a possible way to parallelise the reading is to search into the byte array
  and look for c=63. This is not necessarily the end of a frame though. One
  can advance the read until (dt == 0xFFFFFFFF and x == 0 and y == 0), which
  signifies the next frame starting. You could then note down the filepos of
  that start location. You would then read from that point onwards.
  A different reader would have to read up to where that file location starts.
"""

@cython.boundscheck(False)
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
    (f_events, t_events, y_events, x_events), end_events:
        x_events, y_events, t_events and f_events are numpy arrays containing
        the events. end_events is an array containing the byte offsets to the
        end of the last successful event read from the file. Use this value to
        extract more events from the same file at a future date.
    """
    if max_frames is None:
        max_frames = ii32.max
    cdef int max_framesi = int(max_frames)

    fi = f
    auto_f = None
    if not hasattr(fi, 'read'):
        auto_f = open(f, 'rb')
        fi = auto_f

    cdef:
        Py_ssize_t frame_number = -1
        Py_ssize_t i = 0
        Py_ssize_t num_events = 0
        unsigned int dt = 0
        unsigned int t = 0
        int x = -0
        int y = -0
        int state = 0
        unsigned char c
        int event_ended = 0
        Py_ssize_t filepos = 0
        int bufsize = 524288 * 2
        int bytes_read = 0

        cnp.ndarray[cnp.int32_t, ndim=1] x_events = np.array((), dtype=np.int32)
        cnp.ndarray[cnp.int32_t, ndim=1] y_events = np.array((), dtype=np.int32)
        cnp.ndarray[cnp.uint32_t, ndim=1] t_events = np.array((), dtype=np.uint32)
        cnp.ndarray[cnp.int32_t, ndim=1] f_events = np.array((), dtype=np.int32)
        cnp.ndarray[cnp.uint32_t, ndim=1] end_events = np.array((), dtype=np.uint32)

        # these are buffers to store events from each read of the file
        # use of buffers prevents continual allocation of memory.
        cnp.ndarray[cnp.int32_t, ndim=1] x_neutrons = np.zeros((bufsize), dtype=np.int32)
        cnp.ndarray[cnp.int32_t, ndim=1] y_neutrons = np.zeros((bufsize), dtype=np.int32)
        cnp.ndarray[cnp.uint32_t, ndim=1] t_neutrons = np.zeros((bufsize), dtype=np.uint32)
        cnp.ndarray[cnp.int32_t, ndim=1] f_neutrons = np.zeros((bufsize), dtype=np.int32)
        cnp.ndarray[cnp.uint32_t, ndim=1] end_event_pos = np.zeros((bufsize), dtype=np.uint32)

        int[:] x_neutrons_buf = x_neutrons
        int[:] y_neutrons_buf = y_neutrons
        unsigned int[:] t_neutrons_buf = t_neutrons
        int[:] f_neutrons_buf = f_neutrons
        unsigned int[:] end_event_pos_buf = end_event_pos

        const unsigned char[:] bufv

    buffer = bytearray(bufsize)
    bufv = memoryview(buffer)

    while True and frame_number < max_framesi:
        num_events = 0

        fi.seek(end_last_event + 1)
        bytes_read = fi.readinto(buffer)

        # buffer = fi.read(bufsize)
        # bytes_read = len(buffer)
        # bufv = memoryview(buffer)

        filepos = end_last_event + 1

        if not bytes_read:
            break

        state = 0

        for i in range(bytes_read):
            c = bufv[i]
            if state == 0:
                x = c
                state += 1
            elif state == 1:
                x |= (<unsigned int>(c & 0x3)) << 8;
                if (x & 0x200):
                    x |= <int> 0xFFFFFC00
                y = c >> 2

                state += 1
            else:
                if state == 2:
                    y |= ((<unsigned int> c) & 0xF) << 6;
                    if (y & 0x200):
                        y |= <int> 0xFFFFFC00

                event_ended = (state >= 7 or (c & 0xC0) != 0xC0)

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

                    if dt == <unsigned int> 0xFFFFFFFF and x == 0 and y == 0:
                        t = 0
                        frame_number += 1
                        if frame_number == max_framesi:
                            break
                    else:
                        t += dt
                        if frame_number == -1:
                            return None
                        x_neutrons_buf[num_events] = x
                        y_neutrons_buf[num_events] = y
                        t_neutrons_buf[num_events] = t
                        f_neutrons_buf[num_events] = frame_number
                        end_event_pos_buf[num_events] = end_last_event
                        num_events += 1

        if num_events:
            x_events = np.append(x_events, x_neutrons_buf[0:num_events])
            y_events = np.append(y_events, y_neutrons_buf[0:num_events])
            t_events = np.append(t_events, t_neutrons_buf[0:num_events])
            f_events = np.append(f_events, f_neutrons_buf[0:num_events])
            end_events = np.append(end_events, end_event_pos_buf[0:num_events])

    t_events //= 1000

    if auto_f:
        auto_f.close()
    return (f_events, t_events, y_events, x_events), end_events
