from typing import Text, Dict, Any

from shuai.common import load_data_from_json, TrainerModelConfig, TrainingData
from shuai.engine import Trainer, Runner


def train(data: Dict[Text, Any] = None, model_path: Text = 'models'):
    training_data = load_data_from_json(data)
    cfg = TrainerModelConfig()
    trainer = Trainer(cfg)
    trainer.train(training_data)
    trainer.persist(model_path)


def load(model_path: Text) -> Runner:
    return Runner.load(model_path)


def parse(text: Text, interpreter: Runner) -> Dict:
    return interpreter.parse(text)
