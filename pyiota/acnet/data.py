from typing import Optional


class ACNETResponse:
    def __init__(self, timestamp: Optional[float], t_read: Optional[float] = None):
        self.timestamp = timestamp
        self.t_read = t_read

    def _format_t(self):
        if self.t_read is not None:
            return f'{self.t_read * 1e3:.2f} ms'
        else:
            return ''


class DataResponse(ACNETResponse):
    def __init__(self, data, timestamp, t_read=None):
        super().__init__(timestamp, t_read)
        self.data = data
        self.metadata = {'timestamp': timestamp}

    # def data(self):
    #     return self._data

    def __str__(self):
        return f'{self.__class__.__name__}({self.data=},{self.timestamp=},t_read={self._format_t()})'

class RawDataResponse(DataResponse):
    pass

class DoubleDataResponse(DataResponse):
    pass


class StatusDataResponse(DataResponse):
    def __init__(self, data, timestamp, t_read=None):
        # if 'ready' not in data:
        #     raise Exception(f'Missing ready in {data}')
        super().__init__(data, timestamp, t_read)
        # self.metadata = {'timestamp': timestamp}

    def __str__(self):
        d = self.data
        s = 'StatusDataResponse('
        s += 'READY,' if d['ready'] else 'TRIPPED,'
        s += 'ON,' if d['on'] else 'OFF,'
        s += 'REMOTE,' if d['remote'] else 'REMOTE,'
        s += f't_read={self._format_t()})'
        return s


class ArrayDataResponse(DataResponse):
    pass


# class SettingsResponse(ACNETResponse):
#     pass

class ACNETErrorResponse(ACNETResponse):
    def __init__(self, facility_code: int, error_number: int, message: str,
                 timestamp: Optional[float], t_read: Optional[float] = None
                 ):
        super().__init__(timestamp, t_read)
        self.facility_code = self.fc = facility_code
        self.error_number = self.err = error_number
        self.message = message
        self.metadata = {'timestamp': timestamp}

    @property
    def is_success(self):
        return self.error_number == 0 if self.error_number is not None else False

    @property
    def is_warning(self):
        return self.error_number > 0 if self.error_number is not None else None

    def __str__(self):
        return f'ACNETErrorResponse(fc={self.facility_code},err={self.error_number},msg=' \
               f'\"{self.message}\",t_read={self._format_t()})'

    def __repr__(self):
        return self.__str__()
