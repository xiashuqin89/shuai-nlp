import json
import argparse
from typing import Text

import six

from src.nlp.components import ComponentBuilder
from src.engine.runner.runner import Runner
from cli import Cmd


class RunCmd(Cmd):
    def __init__(self):
        super().__init__()

    def create_argument_parser(self):
        parser = argparse.ArgumentParser(
            description='run a NLU model locally on the command line '
                        'for manual testing')
        parser.add_argument('-m', '--model', required=True,
                            help="path to model")
        parser.add_argument('-t', '--text', required=True,
                            help="path to model")
        return parser

    @classmethod
    def run(cls,
            text: Text,
            model_path: Text,
            component_builder: ComponentBuilder = None) -> Text:
        interpreter = Runner.load(model_path, component_builder)
        text = text.strip()
        if six.PY2:
            text = text.decode("utf-8")
        r = interpreter.parse(text)
        return json.dumps(r, indent=2)

    def console(self) -> Text:
        cmdline_args = self.create_argument_parser()
        return self.run(cmdline_args.text, cmdline_args.model)


if __name__ == '__main__':
    RunCmd().console()
