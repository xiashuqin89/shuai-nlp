from typing import Text


class InvalidConfigError(ValueError):
    def __init__(self, message: Text):
        super(InvalidConfigError, self).__init__(message)


class RequestError(Exception):
    pass
