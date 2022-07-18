import sys
from typing import Text, Dict, Any

from shuai.common import (
    TrainerModelConfig, TrainingData,
    load_config, load_data_from_json
)
from shuai.engine import Trainer, Runner


def train(data: Dict[Text, Any],
          cfg_path: Text = None,
          model_path: Text = 'models') -> Text:
    training_data = load_data_from_json(data)
    if cfg_path:
        cfg = load_config(cfg_path)
    else:
        cfg = TrainerModelConfig()
    trainer = Trainer(cfg)
    trainer.train(training_data)
    dir_name = trainer.persist(model_path)
    sys.stdout.write('Train successfully...\n')
    return dir_name


def load(model_path: Text) -> Runner:
    return Runner.load(model_path)


def parse(text: Text, interpreter: Runner) -> Dict:
    return interpreter.parse(text)
