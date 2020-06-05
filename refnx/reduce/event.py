# -*- coding: utf-8 -*-
"""
Unpack event files
"""
from collections import namedtuple
import numpy as np
from refnx._lib import possibly_open_file

HAVE_CEVENTS = False
try:
    from refnx.reduce._cevent import _cevents

    HAVE_CEVENTS = True
except ImportError:
    pass

ii32 = np.iinfo(np.int32)


def process_event_stream(events, frames, t_bins, y_bins, x_bins):
    """
    Processes the event mode dataset into a histogrammed detector image.

    Parameters
    ----------
    events : tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        the 4-tuple of events (F, T, Y, X)
    frames : sequence of array-like
        each array in the sequence specifies which frames should be included
    t_bins : array_like
        specifies the time bins required in the image
    y_bins : array_like
        specifies the y bins required in the image
    x_bins : array_like
        specifies the x bins required in the image
    Returns
    -------
    detector, frame_count : np.ndarray, np.ndarray
        The new detector images and the number of frames within each image.
        `detector` has shape
        `(len(frames), len(t_bins) - 1, len(y_bins) - 1, len(x_bins) - 1)`.
        `frame_count` says how many frames went into making each frame in the
        detector image

    Notes
    -----
    Every entry in `frames` is clipped to 0 and the maximum frame number in the
    events. Thus, if
    frames = [[-2, -1, 0, 1, 2, 3]], and the maximum frame number is 2, then
    only the 0, 1, 2 frames are included.

    Examples
    ---------
    >>> from refnx.reduce.event import process_event_stream, events
    >>> d = process_event_stream(e, [np.linspace(0, 2308585, 2308586)],
    ...     [0, 40000], np.linspace(110.5, -110.5, 222),
    ...     np.linspace(210.5, -210.5, 422))
    """
    max_frame = np.max(events[0])

    t_events = np.asarray(events).T

    localxbins = np.array(x_bins)
    localybins = np.array(y_bins)
    localtbins = np.sort(np.array(t_bins))
    reversed_x, reversed_y = False, False

    if localxbins[0] > localxbins[-1]:
        localxbins = localxbins[::-1]
        reversed_x = True

    if localybins[0] > localybins[-1]:
        localybins = localybins[::-1]
        reversed_y = True

    # create an (N, T, Y, X) detector image
    detector = np.zeros(
        (len(frames), len(t_bins) - 1, len(y_bins) - 1, len(x_bins) - 1),
        dtype=np.uint64,
    )
    frame_count = np.zeros(len(frames))

    for i, frame in enumerate(frames):
        frame_numbers = np.unique(np.clip(np.asarray(frame), 0, max_frame))
        frame_count[i] = frame_numbers.size

        idxs = np.isin(t_events[:, 0], frame_numbers)
        filtered_events = t_events[idxs]

        detector[i], edge = np.histogramdd(
            filtered_events[:, 1:], bins=(localtbins, localybins, localxbins)
        )
    if reversed_x:
        detector = detector[:, :, :, ::-1]

    if reversed_y:
        detector = detector[:, :, ::-1, :]

    return detector, frame_count


def framebins_to_frames(frame_bins):
    if frame_bins is None:
        return None

    t_frame_bins = np.asarray(frame_bins)

    frames = []
    for idx in range(t_frame_bins.size - 1):
        frames.append(
            np.arange(
                t_frame_bins[idx], t_frame_bins[idx + 1], dtype=np.uint64
            )
        )

    return frames


def events(f, end_last_event=127, max_frames=None):
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
        Stop reading the event file when have read this many frames. If `None`
        then read all frames

    Returns
    -------
    (f_events, t_events, y_events, x_events), end_events:
        x_events, y_events, t_events and f_events are numpy arrays containing
        the events. end_events is an array containing the byte offset to the
        end of the last successful event read from the file. Use this value to
        extract more events from the same file at a future date.
    """
    if HAVE_CEVENTS:
        return _cevents(
            f, end_last_event=end_last_event, max_frames=max_frames
        )
    else:
        raise RuntimeError("Event streaming code not available")
