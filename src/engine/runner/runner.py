import copy
from typing import List, Dict, Text, Any, Optional

from src.nlp.components import Component, ComponentBuilder, validate_requirements
from src.nlp.meta import Metadata
from src.common import Message, UnsupportedModelError, MissingArgumentError
from src.engine import Menu
from src.version import __version__


class Runner(object):
    def __init__(self,
                 pipeline: List[Component],
                 context: Dict[Text, Any],
                 model_metadata: Optional[Metadata] = None):
        self.pipeline = pipeline
        self.context = context if context is not None else {}
        self.model_metadata = model_metadata
        self.default_response = {
            "intent": {"name": "", "confidence": 0.0}, "entities": []
        }

    @staticmethod
    def ensure_model_compatibility(metadata: Metadata):
        from packaging import version

        model_version = metadata.get("version", "0.0.0")
        if version.parse(model_version) != version.parse(__version__):
            raise UnsupportedModelError(f'Only support version {model_version}')

    @staticmethod
    def create(model_metadata: Metadata,
               component_builder: Optional[ComponentBuilder] = None,
               skip_validation: bool = False):
        """
        A Factory based on metadata
        Read the class path and Init object
        Insert to the sorted pipeline
        """
        if component_builder is None:
            component_builder = ComponentBuilder()

        if not skip_validation:
            validate_requirements(model_metadata.component_classes)

        pipeline, context = [], {}
        for component_name in model_metadata.component_classes:
            component = component_builder.load_component(
                component_name, model_metadata.model_dir,
                model_metadata, Menu(), **context)
            try:
                updates = component.provide_context()
                if updates:
                    context.update(updates)
                pipeline.append(component)
            except MissingArgumentError as e:
                raise Exception("Failed to initialize component '{}'. "
                                "{}".format(component.name, e))

        return Runner(pipeline, context, model_metadata)

    @staticmethod
    def load(model_dir: Text,
             component_builder: Optional[ComponentBuilder] = None,
             skip_validation: bool = False):
        """Creates an interpreter based on a persisted model."""

        model_metadata = Metadata.load(model_dir)

        Runner.ensure_model_compatibility(model_metadata)

        return Runner.create(model_metadata,
                             component_builder,
                             skip_validation)

    def parse(self,
              text: Text,
              time=None,
              only_output_properties=True) -> Dict[Text, Any]:
        """
        Parse the input text, classify it and return pipeline result.
        The pipeline result usually contains intent and entities.
        """

        if not text:
            response = self.default_response
            response["text"] = text
            return response

        message = Message(text, self.default_response, time=time)

        for component in self.pipeline:
            component.process(message, **self.context)

        output = copy.deepcopy(self.default_response)
        result = message.as_dict(only_output_properties=only_output_properties)
        output['intent'] = result['intent']
        output['entities'] = result['entities']
        return output
