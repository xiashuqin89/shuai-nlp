import os
import datetime
from collections import defaultdict

from typing import Dict, Text, Any, Optional
from src.common import (
    write_json_to_file, read_json_file, logger,
    InvalidProjectError
)


class Metadata(object):
    def __init__(self, metadata: Dict[Text, Any], model_dir: Optional[Text]):
        self.metadata = metadata or defaultdict()
        self.model_dir = model_dir

    @staticmethod
    def load(model_dir: Text) -> 'Metadata':
        """Loads the metadata from a models directory."""
        try:
            metadata_file = os.path.join(model_dir, 'metadata.json')
            data = read_json_file(metadata_file)
            return Metadata(data, model_dir)
        except InvalidProjectError as e:
            abspath = os.path.abspath(os.path.join(model_dir, 'metadata.json'))
            logger.error(f'Failed to load model metadata from "{abspath}"."{e}"')
            return None

    def get(self, property_name, default=None):
        return self.metadata.get(property_name, default)

    @property
    def language(self) -> Optional[Text]:
        """Language of the underlying model"""
        return self.get('language')

    def persist(self, model_dir: Text):
        """Persists the metadata of a model to a given directory."""

        metadata = self.metadata.copy()
        metadata.update({
            "trained_at": datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
            "version": '',
        })

        filename = os.path.join(model_dir, 'metadata.json')
        write_json_to_file(filename, metadata, indent=4)
