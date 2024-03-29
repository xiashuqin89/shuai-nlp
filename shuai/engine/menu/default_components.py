from shuai.nlp.tokenizers import WhitespaceTokenizer, JiebaTokenizer
from shuai.nlp.featurizers import (
    CountVectorsFeaturizer, TfIdfVectorsFeaturizer, RegexFeaturizer
)
from shuai.nlp.extractors import (
    RegexEntityExtractor, EntitySynonymMapper, RuleEntityExtractor
)
from shuai.nlp.classifiers import (
    KeywordIntentClassifier, TfIdfIntentClassifier, GridSearchIntentClassifier,
    BM25IntentClassifier
)


COMPONENT_CLASSES = [
    WhitespaceTokenizer, JiebaTokenizer,
    CountVectorsFeaturizer, TfIdfVectorsFeaturizer, RegexFeaturizer,
    RegexEntityExtractor, EntitySynonymMapper, RuleEntityExtractor,
    KeywordIntentClassifier, TfIdfIntentClassifier, GridSearchIntentClassifier,
    BM25IntentClassifier
]
