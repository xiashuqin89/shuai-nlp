from typing import Text


class InvalidConfigError(ValueError):
    def __init__(self, message: Text):
        super(InvalidConfigError, self).__init__(message)


class RequestError(Exception):
    pass


class InvalidProjectError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message
