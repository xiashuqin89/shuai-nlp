import abc
import sys
import logging
import argparse
from typing import Text
from importlib import import_module


class Cmd(abc.ABC):
    @abc.abstractmethod
    def create_argument_parser(self):
        pass

    @abc.abstractmethod
    def console(self, *args, **kwargs):
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


class Management:
    def __init__(self, argv=None):
        self.argv = argv or sys.argv[:]

    def execute(self):
        try:
            sub_command = self.argv[1]
        except IndexError:
            sub_command = 'help'

        if sub_command == 'help' or self.argv[1:] in (['--help'], ['-h']):
            sys.stdout.write(f'try to run python control.py train/run\n')
        elif sub_command == 'version' or self.argv[1:] == ['--version']:
            from shuai.version import __version__
            sys.stdout.write(f'{__version__}\n')
        else:
            command = self.fetch_command(sub_command)
            if not command:
                sys.stdout.write('not a known command\n')

    def fetch_command(self, sub_command: Text) -> bool:
        try:
            module = import_module(f'shuai.cli.{sub_command}')
        except ModuleNotFoundError:
            return False
        module.call(self.argv[2:])
        return True
