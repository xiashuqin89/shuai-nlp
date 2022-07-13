TEXT = 'text'
TOKENS = 'tokens'
TEXT_FEATURES = 'text_features'
INTENT_FEATURES = 'intent_features'
INTENT = 'intent'
ENTITIES = 'entities'
RANKING = 'intent_ranking'

TOKENIZER_WHITESPACE = 'WhitespaceTokenizer'
TOKENIZER_JIEPA = 'JiebaTokenizer'

FEATURIZER_REGEX = 'RegexFeaturizer'
FEATURIZER_COUNT_VECTORS = 'CountVectorsFeaturizer'
FEATURIZER_TF_IDF = 'TfIdfVectorsFeaturizer'

CLASSIFIER_KEYWORD = 'KeywordIntentClassifier'
CLASSIFIER_GRID_SEARCH = 'GridSearchIntentClassifier'
CLASSIFIER_TF_IDF = 'TfIdfIntentClassifier'

EXTRACTOR_SYNONYMS = 'EntitySynonymMapper'
EXTRACTOR_CRF = 'CRFEntityExtractor'
EXTRACTOR_REGEX = 'RegexEntityExtractor'

DEFAULT_DICT_FILE_NAME = 'jieba_default_dict'
USER_DICTS_FOLDER_NAME = 'jieba_user_dicts/'
USER_DICT_FILE_NAME = USER_DICTS_FOLDER_NAME + 'user_dict.txt'

FEATURIZER_REGEX_FILE_NAME = 'featurizer_regex.json'

INTENT_RANKING_LENGTH = 5

ENTITY_SYNONYMS_FILE_NAME = 'entity_synonyms.json'
ENTITY_ATTRIBUTE_TYPE = 'entity'
ENTITY_ATTRIBUTE_START = 'start'
ENTITY_ATTRIBUTE_END = 'end'
ENTITY_ATTRIBUTE_VALUE = 'value'

ENTITY_REGEX_FILE_NAME = 'entity_regex.json'
