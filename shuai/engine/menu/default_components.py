from shuai.nlp.tokenizers import WhitespaceTokenizer, JiebaTokenizer
from shuai.nlp.featurizers import (
    CountVectorsFeaturizer, TfIdfVectorsFeaturizer, RegexFeaturizer
)
from shuai.nlp.extractors import RegexEntityExtractor, EntitySynonymMapper
from shuai.nlp.classifiers import (
    KeywordIntentClassifier, TfIdfIntentClassifier, GridSearchIntentClassifier
)


COMPONENT_CLASSES = [
    WhitespaceTokenizer, JiebaTokenizer,
    CountVectorsFeaturizer, TfIdfVectorsFeaturizer, RegexFeaturizer,
    RegexEntityExtractor, EntitySynonymMapper,
    KeywordIntentClassifier, TfIdfIntentClassifier, GridSearchIntentClassifier
]
