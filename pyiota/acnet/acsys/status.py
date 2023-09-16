import warnings

class Status(Exception):
    """An ACSys status type."""

    def __init__(self, val):
        """Creates a status value which is initialized with the supplied
        value. The value must be in the range of signed, 16-bit
        integers.
        """
        if val > -0x8000 and val <= 0x7fff:
            self.value = val
        else:
            raise ValueError('raw status values are 16-bit, signed integers')

    @property
    def facility(self):
        """Returns the 'facility' code of a status value."""
        return self.value & 255

    @property
    def errCode(self):
        """Returns the 'error' code of a status value."""
        warnings.warn(
            "deprecated in favor of the snake_case version, err_code",
            DeprecationWarning)
        return self.err_code

    @property
    def err_code(self):
        """Returns the 'error' code of a status value."""
        return self.value // 256

    @property
    def isSuccess(self):
        """Returns True if the status represents a success status."""
        warnings.warn(
            "deprecated in favor of the snake_case version, is_success",
            DeprecationWarning)
        return self.is_success

    @property
    def is_success(self):
        """Returns True if the status represents a success status."""
        return self.errCode == 0

    @property
    def isFatal(self):
        """Returns True if the status represents a fatal status."""
        warnings.warn(
            "deprecated in favor of the snake_case version, is_fatal",
            DeprecationWarning)
        return self.is_fatal

    @property
    def is_fatal(self):
        """Returns True if the status represents a fatal status."""
        return self.errCode < 0

    @property
    def isWarning(self):
        """Returns True if the status represents a warning status."""
        warnings.warn(
            "deprecated in favor of the snake_case version, is_warning",
            DeprecationWarning)
        return self.is_warning

    @property
    def is_warning(self):
        """Returns True if the status represents a warning status."""
        return self.errCode > 0

    def __eq__(self, other):
        return self.value == other.value \
            if isinstance(other, Status) else False

    def __ne__(self, other):
        return self.value != other.value \
            if isinstance(other, Status) else True

    def __str__(self):
        return '[' + str(self.facility) + ' ' + str(self.errCode) + ']'

# This section defines common ACNET status codes.

ACNET_SUCCESS = Status(1 + 256 * 0)
ACNET_PEND = Status(1 + 256 * 1)
ACNET_ENDMULT = Status(1 + 256 * 2)

ACNET_RETRY = Status(1 + 256 * -1)
ACNET_NOLCLMEM = Status(1 + 256 * -2)
ACNET_NOREMMEM = Status(1 + 256 * -3)
ACNET_RPLYPACK = Status(1 + 256 * -4)
ACNET_REQPACK = Status(1 + 256 * -5)
ACNET_REQTMO = Status(1 + 256 * -6)
ACNET_QUEFULL = Status(1 + 256 * -7)
ACNET_BUSY = Status(1 + 256 * -8)
ACNET_NOT_CONNECTED = Status(1 + 256 * -21)
ACNET_ARG = Status(1 + 256 * -22)
ACNET_IVM = Status(1 + 256 * -23)
ACNET_NO_SUCH = Status(1 + 256 * -24)
ACNET_REQREJ = Status(1 + 256 * -25)
ACNET_CANCELLED = Status(1 + 256 * -26)
ACNET_NAME_IN_USE = Status(1 + 256 * -27)
ACNET_NCR = Status(1 + 256 * -28)
ACNET_NO_NODE = Status(1 + 256 * -30)
ACNET_TRUNC_REQUEST = Status(1 + 256 * -31)
ACNET_TRUNC_REPLY = Status(1 + 256 * -32)
ACNET_NO_TASK = Status(1 + 256 * -33)
ACNET_DISCONNECTED = Status(1 + 256 * -34)
ACNET_LEVEL2 = Status(1 + 256 * -35)
ACNET_HARD_IO = Status(1 + 256 * -41)
ACNET_NODE_DOWN = Status(1 + 256 * -42)
ACNET_SYS = Status(1 + 256 * -43)
ACNET_NXE = Status(1 + 256 * -44)
ACNET_BUG = Status(1 + 256 * -45)
ACNET_NE1 = Status(1 + 256 * -46)
ACNET_NE2 = Status(1 + 256 * -47)
ACNET_NE3 = Status(1 + 256 * -48)
ACNET_UTIME = Status(1 + 256 * -49)
ACNET_INVARG = Status(1 + 256 * -50)
ACNET_MEMFAIL = Status(1 + 256 * -51)
ACNET_NO_HANDLE = Status(1 + 256 * -52)
