import json
import argparse
from typing import Text

import six

from src.nlp.components import ComponentBuilder
from src.engine.runner.runner import Runner
from src.common import logger
from cli import Cmd


class ShellCmd(Cmd):
    def __init__(self):
        super(ShellCmd, self).__init__()

    def create_argument_parser(self):
        parser = argparse.ArgumentParser(
            description='run a NLU model locally on the command line '
                        'for manual testing')
        parser.add_argument('-m', '--model', required=True,
                            help="path to model")
        return parser

    @classmethod
    def run(cls,
            model_path: Text,
            component_builder: ComponentBuilder = None) -> Text:
        interpreter = Runner.load(model_path, component_builder)
        logger.info("NLU model loaded. Type a message and "
                    "press enter to parse it.")
        while True:
            text = input().strip()
            if six.PY2:
                text = text.decode("utf-8")
            r = interpreter.parse(text)
            print(json.dumps(r, indent=2))
            logger.info("Next message:")

    def console(self) -> Text:
        cmdline_args = self.create_argument_parser()
        return self.run(cmdline_args.model)


if __name__ == '__main__':
    ShellCmd().console()
