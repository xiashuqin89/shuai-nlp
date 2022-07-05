from src.nlp.tokenizers import WhitespaceTokenizer, JiebaTokenizer
from src.nlp.featurizers import (
    CountVectorsFeaturizer, TfIdfVectorsFeaturizer, RegexFeaturizer
)
from src.nlp.extractors import RegexEntityExtractor, EntitySynonymMapper
from src.nlp.classifiers import (
    KeywordIntentClassifier, TfIdfIntentClassifier, GridSearchIntentClassifier
)


COMPONENT_CLASSES = [
    WhitespaceTokenizer, JiebaTokenizer,
    CountVectorsFeaturizer, TfIdfVectorsFeaturizer, RegexFeaturizer,
    RegexEntityExtractor, EntitySynonymMapper,
    KeywordIntentClassifier, TfIdfIntentClassifier, GridSearchIntentClassifier
]
