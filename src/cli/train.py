import argparse
from typing import Text, Optional

from src.engine.training import Trainer
from src.common import (
    TrainerModelConfig,
    load_data, load_data_from_url,
    logger, load_config
)
from src.nlp.components import ComponentBuilder
from src.nlp.persistor import get_persistor
from .client import Cmd


class TrainCmd(Cmd):
    def __init__(self):
        super().__init__()

    @classmethod
    def do_train(cls,
                 cfg: TrainerModelConfig,
                 data: Text,
                 path: Optional[Text] = None,
                 project: Optional[Text] = None,
                 fixed_model_name: Optional[Text] = None,
                 storage: Optional[Text] = None,
                 component_builder: Optional[ComponentBuilder] = None,
                 url: Optional[Text] = None,
                 **kwargs):
        trainer = Trainer(cfg, component_builder)
        if url is not None:
            training_data = load_data_from_url(url, cfg.language)
        else:
            training_data = load_data(data, cfg.language)
        trainer.train(training_data, **kwargs)

        if path:
            persistor = get_persistor(storage)
            trainer.persist(path, persistor, project, fixed_model_name)

    def create_argument_parser(self):
        parser = argparse.ArgumentParser(
            description='train a custom language parser')

        parser.add_argument('-o', '--path',
                            default=None,
                            help="Path where model files will be saved")

        group = parser.add_mutually_exclusive_group(required=True)

        group.add_argument('-d', '--data',
                           default=None,
                           help="Location of the training data. For JSON and "
                                "markdown data, this can either be a single file "
                                "or a directory containing multiple training "
                                "data files.")

        group.add_argument('-u', '--url',
                           default=None,
                           help="URL from which to retrieve training data.")

        parser.add_argument('-c', '--config',
                            required=True,
                            help="A json or md file contain intent, entity, "
                                 "synonym, regex")

        parser.add_argument('-t', '--num_threads',
                            default=1,
                            type=int,
                            help="Number of threads to use during model training")

        parser.add_argument('--project',
                            default=None,
                            help="Project this model belongs to.")

        parser.add_argument('--fixed_model_name',
                            help="If present, a model will always be persisted "
                                 "in the specified directory instead of creating "
                                 "a folder like 'model_20171020-160213'")

        parser.add_argument('--storage',
                            help='Set the remote location where models are stored. '
                                 'E.g. on Qcloud Cos. If nothing is configured, the '
                                 'server will only serve the models that are '
                                 'on disk in the configured `path`.')
        self.add_logging_option_arguments(parser)
        return parser

    def console(self):
        cmdline_args = self.create_argument_parser().parse_args()
        self.do_train(load_config(cmdline_args.config),
                      cmdline_args.data,
                      cmdline_args.path,
                      cmdline_args.project,
                      cmdline_args.fixed_model_name,
                      cmdline_args.storage,
                      url=cmdline_args.url,
                      num_threads=cmdline_args.num_threads)
        logger.info("training finished")


if __name__ == '__main__':
    TrainCmd().console()
