class ACNETError(BaseException):
    pass


class ACNETProxyError(ACNETError):
    pass


class ACNETConfigError(ACNETError):
    pass


class ACNETTimeoutError(ACNETError):
    pass


class SequencerError(ACNETError):
    pass
