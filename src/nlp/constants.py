TEXT = 'text'
TOKENS = 'tokens'
TEXT_FEATURES = 'text_features'
INTENT_FEATURES = 'intent_features'
INTENT = 'intent'
ENTITIES = 'entities'
RANKING = 'intent_ranking'

TOKENIZER_WHITESPACE = 'tokenizer_whitespace'
TOKENIZER_JIEPA = 'tokenizer_jieba'

FEATURIZER_REGEX = 'intent_entity_featurizer_regex'
FEATURIZER_COUNT_VECTORS = 'intent_featurizer_count_vectors'
FEATURIZER_TF_IDF = 'intent_featurizer_tf_idf'

CLASSIFIER_KEYWORD = 'intent_classifier_keyword'
CLASSIFIER_GRID_SEARCH = 'intent_classifier_grid_search'
CLASSIFIER_TF_IDF = 'intent_classifier_tf_idf'

EXTRACTOR_SYNONYMS = 'ner_synonyms'
EXTRACTOR_CRF = 'ner_crf'
EXTRACTOR_REGEX = 'RegexEntityExtractor'

DEFAULT_DICT_FILE_NAME = 'jieba_default_dict'
USER_DICTS_FOLDER_NAME = 'jieba_user_dicts/'
USER_DICT_FILE_NAME = USER_DICTS_FOLDER_NAME + 'user_dict.txt'

FEATURIZER_REGEX_FILE_NAME = 'featurizer_regex.json'

INTENT_RANKING_LENGTH = 10

ENTITY_SYNONYMS_FILE_NAME = 'entity_synonyms.json'
ENTITY_ATTRIBUTE_TYPE = 'entity'
ENTITY_ATTRIBUTE_START = 'start'
ENTITY_ATTRIBUTE_END = 'end'
ENTITY_ATTRIBUTE_VALUE = 'value'

ENTITY_REGEX_FILE_NAME = 'entity_regex.json'
