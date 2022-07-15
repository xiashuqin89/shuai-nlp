from typing import Text


class InvalidConfigError(ValueError):
    def __init__(self, message: Text):
        super(InvalidConfigError, self).__init__(message)


class RequestError(Exception):
    pass


class InvalidProjectError(Exception):
    def __init__(self, message: Text):
        self.message = message

    def __str__(self):
        return self.message


class MissingArgumentError(ValueError):
    def __init__(self, message: Text):
        super(MissingArgumentError, self).__init__(message)
        self.message = message

    def __str__(self):
        return self.message


class UnsupportedModelError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class PipelineRunningAbnormalError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class UnsupportedLanguageError(Exception):
    def __init__(self, component: Text, language: Text):
        self.component = component
        self.language = language

        super(UnsupportedLanguageError, self).__init__(component, language)

    def __str__(self):
        return "component {} does not support language {}".format(
            self.component, self.language
        )
