import os
import glob
import shutil

from typing import Text, List, Dict, Any, Optional

from src.nlp.components import Component
from src.common import TrainerModelConfig
from src.nlp.meta import Metadata
from src.nlp.constants import (
    TOKENS, TOKENIZER_JIEPA,
    DEFAULT_DICT_FILE_NAME, USER_DICTS_FOLDER_NAME,
    USER_DICT_FILE_NAME
)
from .tokenizer import Tokenizer, Token


class JiebaTokenizer(Tokenizer, Component):
    name = TOKENIZER_JIEPA
    provides = [TOKENS]
    language_list = ["zh"]

    def __init__(self,
                 component_config: Dict[Text, Any] = None,
                 tokenizer=None):
        super(JiebaTokenizer, self).__init__(component_config)
        self.tokenizer = tokenizer

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["jieba"]

    def tokenize(self, text: Text) -> List[Token]:
        tokenized = self.tokenizer.tokenize(text)
        tokens = [Token(word, start) for (word, start, end) in tokenized]
        return tokens

    @classmethod
    def create(cls, cfg: TrainerModelConfig):
        import jieba as tokenizer
        component_conf = cfg.for_component(cls.name, cls.defaults)
        tokenizer = cls.init_jieba(tokenizer, component_conf)
        return cls(component_conf, tokenizer)

    @classmethod
    def load(cls,
             model_dir: Optional[Text] = None,
             model_metadata: Optional[Metadata] = None,
             cached_component: Optional[Component] = None,
             **kwargs):
        """
        Find corpus path and add to config
        component_meta attribute: name, class, default_dict,
        user_dicts, user_dicts
        """
        import jieba as tokenizer
        component_meta = model_metadata.for_component(cls.name)
        if component_meta.get("default_dict"):
            path_default_dict = os.path.join(model_dir, component_meta.get("default_dict"))
            component_meta["default_dict"] = path_default_dict
        if component_meta.get("user_dicts"):
            path_user_dicts = os.path.join(model_dir, component_meta.get("user_dicts"))
            component_meta["user_dicts"] = path_user_dicts
        tokenizer = cls.init_jieba(tokenizer, component_meta)
        return cls(component_meta, tokenizer)

    @classmethod
    def init_jieba(cls, tokenizer, dict_config: Dict[Text, Text]):
        if dict_config.get("default_dict") and os.path.isfile(dict_config.get("default_dict")):
            path_default_dict = glob.glob("{}".format(dict_config.get("default_dict")))
            tokenizer = cls.set_default_dict(tokenizer, path_default_dict[0])
        if dict_config.get("user_dicts"):
            if os.path.isdir(dict_config.get("user_dicts")):
                parse_pattern = "{}/*"
            else:
                parse_pattern = "{}"
            path_user_dicts = glob.glob(parse_pattern.format(dict_config.get("user_dicts")))
            tokenizer = cls.set_user_dicts(tokenizer, path_user_dicts)
        return tokenizer

    @staticmethod
    def set_default_dict(tokenizer, path_default_dict):
        """Set jieba dictionary"""
        tokenizer.set_dictionary(path_default_dict)
        return tokenizer

    @staticmethod
    def set_user_dicts(tokenizer, path_user_dicts):
        """Load user dictionary"""
        if len(path_user_dicts) > 0:
            for path_user_dict in path_user_dicts:
                tokenizer.load_userdict(path_user_dict)
        return tokenizer

    def persist(self, model_dir: Text) -> Dict[Text, Any]:
        return_dict = {}
        if self.component_config.get("default_dict"):
            des_path_default_dict = os.path.join(model_dir, DEFAULT_DICT_FILE_NAME)
            if os.path.isfile(self.component_config.get("default_dict")):
                shutil.copy2(self.component_config.get("default_dict"), des_path_default_dict)
                return_dict.update({"default_dict": DEFAULT_DICT_FILE_NAME})

        if self.component_config.get("user_dicts"):
            des_path_user_dicts = os.path.join(model_dir, USER_DICTS_FOLDER_NAME)
            os.mkdir(des_path_user_dicts)
            if os.path.isdir(self.component_config.get("user_dicts")):
                parse_pattern = "{}/*"
                path_user_dicts = glob.glob(parse_pattern.format(self.component_config.get("user_dicts")))
                for path_user_dict in path_user_dicts:
                    shutil.copy2(path_user_dict, des_path_user_dicts)
                return_dict.update({"user_dicts": USER_DICTS_FOLDER_NAME})
            else:
                des_path_user_dict = os.path.join(model_dir, USER_DICT_FILE_NAME)
                shutil.copy2(self.component_config.get("user_dicts"), des_path_user_dict)
                return_dict.update({"user_dicts": USER_DICT_FILE_NAME})

        return return_dict
