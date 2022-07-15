from shuai.nlp.constants import ENTITIES, TOKENS, EXTRACTOR_CRF
from .extractor import EntityExtractor


class CRFEntityExtractor(EntityExtractor):
    name = EXTRACTOR_CRF
    provides = [ENTITIES]
    requires = [TOKENS]

    @classmethod
    def required_packages(cls):
        return ["sklearn_crfsuite", "sklearn"]
