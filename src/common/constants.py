DEFAULT_CONFIG_LOCATION = "config.yml"
DEFAULT_CONFIG = {
    "language": "en",
    "pipeline": [],
    "data": None,
}
# provide system default pipeline config
# now config is not real config
REGISTERED_PIPELINE_TEMPLATES = {
    "spacy_sklearn": [
        "nlp_spacy",
        "tokenizer_spacy",
        "intent_featurizer_spacy",
        "intent_entity_featurizer_regex",
        "ner_crf",
        "ner_synonyms",
        "intent_classifier_sklearn",
    ],
    "keyword": [
        "intent_classifier_keyword",
    ],
    "tensorflow_embedding": [
        "intent_featurizer_count_vectors",
        "intent_classifier_tensorflow_embedding"
    ]
}
DEFAULT_PROJECT_NAME = "default"

MARKDOWN_INTENT = "intent"
MARKDOWN_SYNONYM = "synonym"
MARKDOWN_REGEX = "regex"
MARKDOWN_AVAILABLE_SECTIONS = [
    MARKDOWN_INTENT,
    MARKDOWN_SYNONYM,
    MARKDOWN_REGEX
]
