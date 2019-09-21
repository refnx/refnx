%%cython -a

# cython: language_level=3, cdivision=False
from collections import namedtuple
from refnx._lib import possibly_open_file
import numpy as np

from libc.stdint cimport uint32_t, int64_t, int32_t, uint8_t
cimport numpy as cnp
cimport cython

import mmap

ii32 = np.iinfo(np.int32)


EventFileHeader_Base = namedtuple('EventFileHeader_Base',
                                  ['magic_number', 'format_number',
                                   'anstohm_version', 'pack_format',
                                   'oob_enabled', 'clock_scale'])
EventFileHeader_Packed = namedtuple('EventFileHeader_Packed',
                                    ['evt_stg_nbits_x', 'evt_stg_nbits_y',
                                     'evt_stg_nbits_v', 'evt_stg_nbits_w',
                                      'evt_stg_nbits_wa', 'evt_stg_xy_signed'])

def event_header(f):
    """
    Reads the header from an ANSTO event file

    Parameters
    ----------
    f : str or file-like
        The event file of interest

    Returns
    -------
    base, packed : EventFileHeader_Base, EventFileHeader_Packed
    """
    with possibly_open_file(f, 'rb') as fh:
        fh.seek(0)
        header = fh.read(128)
        header_arr = np.frombuffer(header, dtype='int32')

    base = EventFileHeader_Base(magic_number=header_arr[0],
                                format_number=header_arr[1],
                                anstohm_version = header_arr[2],
                                pack_format = header_arr[3],
                                oob_enabled = header_arr[4],
                                clock_scale = header_arr[5])
    assert(base.magic_number == 0x0DAE0DAE)

    if header_arr[1] >= 0x00010002:
        evt_stg_nbits_wa = header_arr[20]
        evt_stg_xy_signed = header_arr[21]
    else:
        evt_stg_nbits_wa = 0
        evt_stg_xy_signed = header_arr[20]

    packed = EventFileHeader_Packed(evt_stg_nbits_x=header_arr[16],
                            evt_stg_nbits_y=header_arr[17],
                            evt_stg_nbits_v=header_arr[18],
                            evt_stg_nbits_w=header_arr[19],
                            evt_stg_nbits_wa=evt_stg_nbits_wa,
                            evt_stg_xy_signed=evt_stg_xy_signed)
    return base, packed


"""
event decoding state machine
for all events
initial state - then DECODE_VAL_BITFIELDS (for neutron
events) or DECODE_OOB_BYTE_1 (for OOB events) for OOB events only
"""
cdef int DECODE_START = 0
cdef int DECODE_OOB_BYTE_1 = 1
cdef int DECODE_OOB_BYTE_2 = 2
# for all events
cdef int DECODE_VAL_BITFIELDS = 3
# final state - then output data and return to DECODE_START
cdef int DECODE_DT = 4

# all events contain some or all of these fields
# x, y, v, w, wa
DEF NVAL = 5

"""
Types of OOB events, and 'NEUTRON' event.  Not all are used for all
instruments, or supported yet.

NEUTRON = 0 = a neutron detected, FRAME_START = -2 = T0 pulse (e.g. from
chopper, or from Doppler on Emu).  For most instruments, these are the only
types used.

FRAME_AUX_START = -3 (e.g. from reflecting chopper on Emu), VETO = -6 (e.g.
veto signal from ancillary)

BEAM_MONITOR = -7 (e.g. if beam monitors connected direct to Mesytec MCPD8 DAE)

RAW = -8 = pass-through, non-decoded raw event directly from the DAE (e.g.
Mesytec MCPD8).  Used to access special features of DAE.

Other types are not used in general (DATASIZES = -1 TBD in future, FLUSH = -4
deprecated, FRAME_DEASSERT = -5 only on Fastcomtec P7888 DAE).
"""

@cython.boundscheck(False)
@cython.cdivision(False)
def events(f, def_clock_scale=1000, use_tx_chopper=False):

    # read file header (base header then packed-format header)
    hdr_base, hdr_packed = event_header(f)

    if hdr_base.pack_format:
        raise RuntimeError("only packed binary format is supported")

    # data starts at byte 128
    f.seek(128)

    """
    Setup the clock_scale.  In format 0x00010001 this was not part of the
    headers, hence a function argument is provided to allow it to be
    specified manually. In the current format 0x00010002, clock_scale is
    written to the header and need not be specified, unless some alternate
    scale is needed.
    """
    scale_microsec = 1 / hdr_base.clock_scale
    if not hdr_base.clock_scale:
        # old eventfile format did not have clock_scale...
        scale_microsec = 1 / def_clock_scale

    # the initial time is not set correctly so wait until primary and auxillary
    # time have been reset before sending events
    cdef:
        int64_t primary_time = 0
        int64_t auxillary_time = 0
        bint primary_ok = False
        bint auxillary_ok = False
        bint evt_stg_xy_signed = hdr_packed.evt_stg_xy_signed

    cdef:
        # event data fields
        uint32_t x = 0, y = 0, v = 0, w = 0, wa = 0
        uint32_t ptr_val[NVAL]
        Py_ssize_t frame_number = -1
        int32_t signed_x, signed_y

    ptr_val = [0, 0, 0, 0, 0]

    """
    All events are also timestamped.  The differential timestamp dt stored in
    each event is summed to recover the event timestamp t. All timestamps are
    frame-relative, i.e. FRAME_START event represents T0 (e.g. from a chopper)
    and t is reset to 0. In OOB mode and for certain DAE types only (e.g.
    Mesytec MCPD8), the FRAME_START event is timestamped relative to the last
    FRAME_START. The timestamp t on the FRAME_START event is therefore the
    total frame duration, and this can be used to recover the absolute
    timestamp of all events in the DAQ, if desired (e.g. for accurate timing
    during long term kinematic experiments).
    """
    # t = 0 dt may be negative occasionally for some DAE types, therefore dt
    # and t are signed ints.
    cdef:
        int32_t dt
        int32_t ind_val = 0
        int32_t nbits_val = 0
        int32_t nbits_val_filled = 0
        int32_t nbits_dt_filled = 0
        int32_t nbits_val_to_fill
        int32_t nbits_ch_used = 0
        uint8_t ch
        bint frame_start_event
        bint event_ended = False
        int32_t oob_en
        int32_t oob_event = 0
        int32_t c = 0
        int32_t nbits_val_neutron[NVAL]
        int32_t nbits_val_oob[NVAL]
        int state = DECODE_START
        int num_events = 0

    nbits_val_oob = [0, 0, 0, 0, 0]
    nbits_val_neutron = [
        hdr_packed.evt_stg_nbits_x,
        hdr_packed.evt_stg_nbits_y,
        hdr_packed.evt_stg_nbits_v,
        hdr_packed.evt_stg_nbits_w,
        hdr_packed.evt_stg_nbits_wa]

    # will be 1 if we are reading a new OOB
    # event file (format 0x00010002 only).
    oob_en = hdr_base.oob_enabled

    # For neutron events, oob_event = 0, and for OOB events,
    # oob_event = 1 and c indicates the OOB event type.
    # c < 0 for all OOB events currently.
    oob_event = 0
    c = 0

    # event decoding state machine
    state = DECODE_START

    while True:
        # read next byte
        www = f.read(1)
        if len(www):
            ch = www[0]
        else:
            break
        # no bits used initially, 8 to go
        nbits_ch_used = 0

        # start of event processing
        if (state == DECODE_START):
            # if OOB event mode is enabled, the leading Bit 0 of the first byte
            # indicates whether the event is a neutron event or an OOB event
            if not oob_en:
                state = DECODE_VAL_BITFIELDS
            else:
                oob_event = ch & 1
                # leading bit used as OOB bit
                nbits_ch_used = 1

                if not oob_event:
                    state = DECODE_VAL_BITFIELDS
                else:
                    state = DECODE_OOB_BYTE_1

            # setup to decode new event bitfields (for both neutron
            # and OOB events)
            for ind_val in range(NVAL):
                ptr_val[ind_val] = 0

            ind_val = 0
            nbits_val_filled = 0

            dt = 0
            nbits_dt_filled = 0

        # state machine for event decoding
        if state == DECODE_START:
            raise RuntimeError('Failure in event decoding')
        elif state == DECODE_OOB_BYTE_1:
            """
            first OOB header byte
            OOB event Byte 1:  Bit 0 = 1 = OOB event, Bit 1 =
            mode (only mode=0 suported currently), Bits 2-5 =
            c (OOB event type), Bits 6-7 = bitfieldsize_x
            / 8. bitfieldsize_x and following 2-bit
            bitfieldsizes are the number of bytes used to
            store the OOB parameter. All of x,y,v,w,wa are
            short integers (16 bits maximum) and so
            bitfieldsizes = 0, 1 or 2 only.
            """
            # Bits 2-5 = c
            c = ch >> 2 & 0xF

            if c & 0x8:
                # c is a signed parameter so sign extend - OOB
                # events are negative values
                c |= <int32_t> 0xFFFFFFF0
            nbits_val_oob[0] = (ch & 0xC0) >> 3  # Bits 6-7 * 8 = bitfieldsize_x

            # Proceed to process second OOB event header
            # byte next time
            state = DECODE_OOB_BYTE_2
        elif state == DECODE_OOB_BYTE_2:
            # second OOB header byte
            # bitfieldsizes for y, v, w and wa, as for
            # bitfieldsize_x above.
            nbits_val_oob[1] = (ch & 0x03) << 3  # Bits 0-1 * 8 = bitfieldsize_y
            nbits_val_oob[2] = (ch & 0x0C) << 1  # Bits 2-3 * 8 = bitfieldsize_v
            nbits_val_oob[3] = (ch & 0x30) >> 1  # Bits 4-5 * 8 = bitfieldsize_w
            nbits_val_oob[4] = (ch & 0xC0) >> 3  # Bits 6-7 * 8 = bitfieldsize_wa

            # Proceed to read and store x, y, v, w, wa for
            # the OOB event
            state = DECODE_VAL_BITFIELDS
        elif state == DECODE_VAL_BITFIELDS:
            # fill bits of the incoming ch to the event's bitfields.
            # stop when we've filled them all, or all bits of ch are used.
            while (ind_val < NVAL) and (nbits_ch_used < 8):
                nbits_val = nbits_val_oob[ind_val]
                if not oob_event:
                    nbits_val = nbits_val_neutron[ind_val]
                if not nbits_val:
                    nbits_val_filled = 0
                    ind_val += 1
                else:
                    nbits_val_to_fill = nbits_val - nbits_val_filled
                    if (8 - nbits_ch_used) >= nbits_val_to_fill:
                        ptr_val[ind_val] |= (((ch >> nbits_ch_used) &
                                              ((1 << nbits_val_to_fill) - 1))
                                               << nbits_val_filled)
                        nbits_val_filled = 0
                        nbits_ch_used += nbits_val_to_fill
                        ind_val += 1
                    else:
                        ptr_val[ind_val] |= (ch >> nbits_ch_used) << nbits_val_filled
                        nbits_val_filled += (8 - nbits_ch_used)
                        nbits_ch_used = 8

            if ind_val == NVAL:
                # and fall through for dt processing
                state = DECODE_DT

            if nbits_ch_used == 8:
                # read next byte
                continue

        if state == DECODE_DT:
            if 8 - nbits_ch_used <= 2:
                dt |= (ch >> nbits_ch_used) << nbits_dt_filled
                nbits_dt_filled += 8 - nbits_ch_used
            elif (ch & 0xC0) == 0xC0:
                dt |= ((ch & 0x3F) >> nbits_ch_used) << nbits_dt_filled
                nbits_dt_filled += (6 - nbits_ch_used)
            else:
                dt |= (ch >> nbits_ch_used) << nbits_dt_filled
                nbits_dt_filled += (8 - nbits_ch_used)
                event_ended = True

        # unpack values from array
        # this would be better done by using a pointer array
        x, y, v, w, wa = ptr_val

        if event_ended:
            # start on new event next time
            state = DECODE_START
            # update times
            primary_time += dt
            auxillary_time += dt

            # is this event a frame_start?
            # FRAME_START is an OOB event when oob
            # mode enabled
            if oob_en:
                frame_start_event = oob_event and c == -2
            else:
                frame_start_event = (x == 0 and y == 0 and dt == -1)

            if oob_en or (not frame_start_event):
                if oob_event:
                    if c == -3:
                        # FRAME_AUX_START = -3
                        # 0 is the reflecting chopper and 1 is the transmission chopper
                        if not use_tx_chopper and x == 0:
                            auxillary_time = 0
                            auxillary_ok = True
                        if use_tx_chopper and x == 1:
                            auxillary_time = 0
                            auxillary_ok = True
                else:
                    # if times are ok, time units in usec
                    if primary_ok and auxillary_ok:
                        if evt_stg_xy_signed:
                            # if x and y are signed then convert from uint32 to int32
                            signed_x, signed_y = x, y
                            if x & (2**(nbits_val_neutron[0] - 1)):
                                signed_x = - (<uint32_t>0x100000000 -
                                              (x | <uint32_t>0xFFFFFC00))
                            if y & (2**(nbits_val_neutron[1] - 1)):
                                signed_y = - (<uint32_t>0x100000000 -
                                              (y | <uint32_t>0xFFFFFC00))
                        # TODO ADD AN EVENT TO A LIST SOMEWHERE
                        print(signed_x, signed_y, primary_time * scale_microsec)
                        num_events += 1

            if frame_start_event:
                # reset timestamp at start of a new frame
                # the auxillary time is only available in OOB mode
                # otherwise, auxillary time = primary time
                primary_time = 0
                primary_ok = True
                if not oob_en:
                    auxillary_time = 0
                    auxillary_ok = True

                # TODO INCREMENT A FRAME NUMBER
                frame_number += 1

            # TODO UPDATE FILE POSITION
            # progress.update(loader.selected_position())

            event_ended = False
    return num_events, frame_number
