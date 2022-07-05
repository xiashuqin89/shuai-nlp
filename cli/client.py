import abc
import logging
import argparse


class Cmd(abc.ABC):
    @abc.abstractmethod
    def create_argument_parser(self):
        pass

    @abc.abstractmethod
    def console(self):
        pass

    @classmethod
    def add_logging_option_arguments(cls, parser: argparse.ArgumentParser, default=logging.WARNING):
        """Add options to an argument parser to configure logging levels."""
        parser.add_argument(
            '--debug',
            help="Print lots of debugging statements. "
                 "Sets logging level to DEBUG",
            action="store_const",
            dest="loglevel",
            const=logging.DEBUG,
            default=default,
        )
        parser.add_argument(
            '-v', '--verbose',
            help="Be verbose. Sets logging level to INFO",
            action="store_const",
            dest="loglevel",
            const=logging.INFO,
        )
