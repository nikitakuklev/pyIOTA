# Generated by the protocol compiler version 1.3.9
# DO NOT EDIT THIS FILE DIRECTLY!

__doc__ = 'Message serializer for the Syncd protocol.'

from itertools import chain, islice

__all__ = ['ProtocolError',
           'unmarshal_request',
           'unmarshal_reply',
           'EvState_struct',
           'EvClock_struct',
           'Event_struct',
           'Discover_request',
           'Register_request',
           'Instance_reply',
           'Report_reply']

class ProtocolError(Exception):
    """Exception class that gets raised when there's a problem marshaling
       or unmarshaling a message from the Syncd protocol."""

    def __init__(self, reason):
        self.reason = reason

    def __str__(self):
        return repr(self.reason)

# -- Internal marshalling routines --

def emitRawInt(tag, val):
    def emitEach(buf, n):
        curr = (val >> (n * 8)) & 0xff
        next = val >> ((n + 1) * 8)
        if (next == 0 and (curr & 0x80) != 0x80) or \
           (next == -1 and (curr & 0x80) == 0x80):
            buf.append(tag + n + 1)
        else:
            emitEach(buf, n + 1)
        buf.append(curr)
    tmp = bytearray()
    emitEach(tmp, 0)
    return tmp

def marshal_int16(val):
    if isinstance(val, int):
        if val < 32768 and val > -32769:
            return emitRawInt(0x10, val)
        else:
            raise ProtocolError('value out of range for int16')
    else:
        raise ProtocolError('expected integer type')

def marshal_int32(val):
    if isinstance(val, int):
        if int(-2147483648) <= val <= int(2147483647):
            return emitRawInt(0x10, val)
        else:
            raise ProtocolError('value out of range for int32')
    else:
        raise ProtocolError('expected integer type')

def marshal_int64(val):
    if isinstance(val, int):
        if int(-9223372036854775808) <= val <= int(9223372036854775807):
            return emitRawInt(0x10, val)
        else:
            raise ProtocolError('value out of range for int64')
    else:
        raise ProtocolError('expected integer type')

def marshal_string(val):
    if isinstance(val, str):
        return chain(emitRawInt(0x40, len(val)),\
                     (ord(ii) for ii in val))
    else:
        raise ProtocolError('expected string type')

def marshal_array(fn, val):
    if isinstance(val, list):
        return chain(emitRawInt(0x50, len(val)),\
                     chain.from_iterable((fn(v) for v in val)))
    else:
        raise ProtocolError('expected list type')

__all__.append('Clock_Tclk')
Clock_Tclk = 8267
__all__.append('Clock_Test')
Clock_Test = 26129
__all__.append('Clock_CMTF')
Clock_CMTF = -27901
__all__.append('Clock_NML')
Clock_NML = 24689

def marshal_Clock_enum(val):
    if int(val) == 8267:
        return b'\x82\x20\x4b'
    elif int(val) == 26129:
        return b'\x82\x66\x11'
    elif int(val) == -27901:
        return b'\x82\x93\x03'
    elif int(val) == 24689:
        return b'\x82\x60\x71'
    else:
        raise ProtocolError("invalid value for enum 'Clock'")

class EvState_struct:
    def __init__(self):
        self.device_index = int(0)
        self.value = int(0)

    def __eq__(self, other):
        return self.device_index == other.device_index and \
            self.value == other.value

    def __ne__(self, other):
        return not self.__eq__(other)

def marshal_EvState_struct(val):
    return chain(b'\x51\x04\x12\x05\xad',
                 marshal_int32(val.device_index),
                 b'\x12\xc1\x60',
                 marshal_int32(val.value))

class EvClock_struct:
    def __init__(self):
        self.event = int(0)
        self.number = int(0)

    def __eq__(self, other):
        return self.event == other.event and \
            self.number == other.number

    def __ne__(self, other):
        return not self.__eq__(other)

def marshal_EvClock_struct(val):
    return chain(b'\x51\x04\x12\x63\x90',
                 marshal_int16(val.event),
                 b'\x12\x24\x8a',
                 marshal_int32(val.number))

class Event_struct:
    def __init__(self):
        self.stamp = int(0)

    def __eq__(self, other):
        return self.stamp == other.stamp and \
            ((not hasattr(self, 'state') and not hasattr(other, 'state')) or \
            (hasattr(self, 'state') and hasattr(other, 'state') and \
             self.state == other.state)) and \
            ((not hasattr(self, 'clock') and not hasattr(other, 'clock')) or \
            (hasattr(self, 'clock') and hasattr(other, 'clock') and \
             self.clock == other.clock))

    def __ne__(self, other):
        return not self.__eq__(other)

def marshal_Event_struct(val):
    return chain(emitRawInt(0x50, 2 \
                    + (2 if hasattr(val, 'state') else 0) \
                    + (2 if hasattr(val, 'clock') else 0)),
                 b'\x12\xc7\x8d',
                 marshal_int64(val.stamp),
                 chain(b'\x12\x9e\x2e',
                       marshal_EvState_struct(val.state)) \
                       if hasattr(val, 'state') else bytearray(),
                 chain(b'\x12\x1b\x19',
                       marshal_EvClock_struct(val.clock)) \
                       if hasattr(val, 'clock') else bytearray())

class Discover_request:
    def __init__(self):
        self.clock = Clock_Tclk

    def __eq__(self, other):
        return self.clock == other.clock

    def __ne__(self, other):
        return not self.__eq__(other)

    def marshal(self):
        """Returns a generator that emits a character stream representing
           the marshaled contents of Discover_request."""
        return chain(b'SDD\x02\x51\x03\x14\xc7\xf5\x7b\x8d\x12\x97\x3c\x51\x02\x12\x1b\x19',
                     marshal_Clock_enum(self.clock))

class Register_request:
    def __init__(self):
        self.evTclk = []

    def __eq__(self, other):
        return self.evTclk == other.evTclk

    def __ne__(self, other):
        return not self.__eq__(other)

    def marshal(self):
        """Returns a generator that emits a character stream representing
           the marshaled contents of Register_request."""
        return chain(b'SDD\x02\x51\x03\x14\xc7\xf5\x7b\x8d\x12\x12\xd6\x51\x02\x12\x99\xdf',
                     marshal_array(marshal_string, self.evTclk))

class Instance_reply:
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False

    def marshal(self):
        """Returns a generator that emits a character stream representing
           the marshaled contents of Instance_reply."""
        return b'SDD\x02\x51\x03\x14\xc7\xf5\x7b\x8d\x12\xb7\x31\x51\x00'

class Report_reply:
    def __init__(self):
        self.seq = int(0)
        self.events = []

    def __eq__(self, other):
        return self.seq == other.seq and \
            self.events == other.events

    def __ne__(self, other):
        return not self.__eq__(other)

    def marshal(self):
        """Returns a generator that emits a character stream representing
           the marshaled contents of Report_reply."""
        return chain(b'SDD\x02\x51\x03\x14\xc7\xf5\x7b\x8d\x12\xed\x8e\x51\x04\x12\xc2\xde',
                     marshal_int16(self.seq),
                     b'\x12\x8b\x06',
                     marshal_array(marshal_Event_struct, self.events))

def marshal_request(val):
    return val.marshal()

def marshal_reply(val):
    return val.marshal()

# -- Internal unmarshalling routines --

def consumeRawInt(ii, tag):
    iiTag = (ii.__next__())
    iiLen = iiTag & 0xf
    if (iiTag & 0xf0) == (tag & 0xf0) and iiLen > 0 and iiLen <= 8:
        firstByte = (ii.__next__())
        retVal = (0 if (0x80 & firstByte) == 0 else -256) | firstByte
        while iiLen > 1:
            retVal = (retVal << 8) | (ii.__next__())
            iiLen = iiLen - 1
        return int(retVal)
    else:
        raise ProtocolError('bad tag or length')

def unmarshal_int16(ii):
    val = consumeRawInt(ii, 0x10)
    if val >= -0x8000 and val < 0x8000:
        return int(val)
    else:
        raise ProtocolError('value out of range for int16')

def unmarshal_int32(ii):
    val = consumeRawInt(ii, 0x10)
    if int(-2147483648) <= val <= int(2147483647):
        return int(val)
    else:
        raise ProtocolError('value out of range for int32')

def unmarshal_int64(ii):
    val = consumeRawInt(ii, 0x10)
    if int(-9223372036854775808) <= val <= int(9223372036854775807):
        return val
    else:
        raise ProtocolError('value out of range for int64')

def unmarshal_string(ii):
    return bytearray(islice(ii, consumeRawInt(ii, 0x40))).decode('utf-8')

def unmarshal_array(ii, fn):
    return [fn(ii) for x in range(consumeRawInt(ii, 0x50))]

def unmarshal_header(ii):
    if ii.__next__() != 83 or ii.__next__() != 68 or \
       ii.__next__() != 68 or ii.__next__() != 2 or \
       consumeRawInt(ii, 0x50) != 3:
        raise ProtocolError('invalid header')
    elif consumeRawInt(ii, 0x10) != -940213363:
        raise ProtocolError('incorrect protocol specified')

def unmarshal_Clock_enum(ii):
    val = consumeRawInt(ii, 0x80)
    if val == 8267:
        return Clock_Tclk
    elif val == 26129:
        return Clock_Test
    elif val == -27901:
        return Clock_CMTF
    elif val == 24689:
        return Clock_NML
    else:
        raise ProtocolError("invalid value for enum 'Clock'")

def unmarshal_EvState_struct(ii):
    nFlds = consumeRawInt(ii, 0x50)
    if nFlds != 4:
        raise ProtocolError('incorrect number of fields')
    else:
        tmp = EvState_struct()
        for xx in range(nFlds // 2):
            fld = consumeRawInt(ii, 0x10)
            if fld == 1453:
                tmp.device_index = unmarshal_int32(ii)
            elif fld == -16032:
                tmp.value = unmarshal_int32(ii)
            else:
                raise ProtocolError('unknown field found')
        return tmp

def unmarshal_EvClock_struct(ii):
    nFlds = consumeRawInt(ii, 0x50)
    if nFlds != 4:
        raise ProtocolError('incorrect number of fields')
    else:
        tmp = EvClock_struct()
        for xx in range(nFlds // 2):
            fld = consumeRawInt(ii, 0x10)
            if fld == 25488:
                tmp.event = unmarshal_int16(ii)
            elif fld == 9354:
                tmp.number = unmarshal_int32(ii)
            else:
                raise ProtocolError('unknown field found')
        return tmp

def unmarshal_Event_struct(ii):
    nFlds = consumeRawInt(ii, 0x50)
    if (nFlds % 2) != 0 or nFlds < 2 or nFlds > 6:
        raise ProtocolError('incorrect number of fields')
    else:
        tmp = Event_struct()
        for xx in range(nFlds // 2):
            fld = consumeRawInt(ii, 0x10)
            if fld == -14451:
                tmp.stamp = unmarshal_int64(ii)
            elif fld == -25042:
                tmp.state = unmarshal_EvState_struct(ii)
            elif fld == 6937:
                tmp.clock = unmarshal_EvClock_struct(ii)
            else:
                raise ProtocolError('unknown field found')
        return tmp

def unmarshal_Discover_request(ii):
    nFlds = consumeRawInt(ii, 0x50)
    if nFlds != 2:
        raise ProtocolError('incorrect number of fields')
    else:
        tmp = Discover_request()
        for xx in range(nFlds // 2):
            fld = consumeRawInt(ii, 0x10)
            if fld == 6937:
                tmp.clock = unmarshal_Clock_enum(ii)
            else:
                raise ProtocolError('unknown field found')
        return tmp

def unmarshal_Register_request(ii):
    nFlds = consumeRawInt(ii, 0x50)
    if nFlds != 2:
        raise ProtocolError('incorrect number of fields')
    else:
        tmp = Register_request()
        for xx in range(nFlds // 2):
            fld = consumeRawInt(ii, 0x10)
            if fld == -26145:
                tmp.evTclk = unmarshal_array(ii, unmarshal_string)
            else:
                raise ProtocolError('unknown field found')
        return tmp

def unmarshal_Instance_reply(ii):
    if consumeRawInt(ii, 0x50) != 0:
        raise ProtocolError('incorrect number of fields')
    else:
        return Instance_reply()

def unmarshal_Report_reply(ii):
    nFlds = consumeRawInt(ii, 0x50)
    if nFlds != 4:
        raise ProtocolError('incorrect number of fields')
    else:
        tmp = Report_reply()
        for xx in range(nFlds // 2):
            fld = consumeRawInt(ii, 0x10)
            if fld == -15650:
                tmp.seq = unmarshal_int16(ii)
            elif fld == -29946:
                tmp.events = unmarshal_array(ii, unmarshal_Event_struct)
            else:
                raise ProtocolError('unknown field found')
        return tmp

def unmarshal_request(ii):
    """Attempts to unmarshal a request message from the specified
       generator, ii. If an error occurs, the ProtocolError exception
       will be raised."""
    try:
        unmarshal_header(ii)
        msg = consumeRawInt(ii, 0x10)
        if msg == -26820:
            return unmarshal_Discover_request(ii)
        elif msg == 4822:
            return unmarshal_Register_request(ii)
        else:
            raise ProtocolError('unknown request type')
    except StopIteration:
        raise ProtocolError('unexpected end of input')

def unmarshal_reply(ii):
    """Attempts to unmarshal a reply message from the specified
       generator, ii. If an error occurs, the ProtocolError exception
       will be raised."""
    try:
        unmarshal_header(ii)
        msg = consumeRawInt(ii, 0x10)
        if msg == -18639:
            return unmarshal_Instance_reply(ii)
        elif msg == -4722:
            return unmarshal_Report_reply(ii)
        else:
            raise ProtocolError('unknown reply type')
    except StopIteration:
        raise ProtocolError('unexpected end of input')
