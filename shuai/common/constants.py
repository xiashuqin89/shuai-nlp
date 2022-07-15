import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_LOCATION = "default.yml"
DEFAULT_CONFIG = {
    'language': 'en',
    'pipeline': [
        {
            'name': 'JiebaTokenizer',
            'user_dicts': os.path.join(BASE_DIR, '../corpus/user_dicts')
        },
        {
            'name': 'TfIdfVectorsFeaturizer',
            'stop_words': os.path.join(BASE_DIR, '../corpus/stopwords.txt')
        },
        {
            'name': 'TfIdfIntentClassifier'
        },
        {
            'name': 'EntitySynonymMapper'
        },
        {
            'name': 'RegexEntityExtractor'
        }
    ]
}
NULL_CONFIG = {
    "language": "en",
    "pipeline": [],
    "data": None
}

# provide system default pipeline config
# now config is not real config
REGISTERED_PIPELINE_TEMPLATES = {

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
