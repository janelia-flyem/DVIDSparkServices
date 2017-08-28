from __future__ import print_function, absolute_import
from jsonschema import Draft4Validator, validators

def extend_with_default(validator_class):
    """
    This code was adapted from the jsonschema FAQ:
    http://python-jsonschema.readthedocs.org/en/latest/faq/
    """
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults_and_validate(validator, properties, instance, schema):
        for property, subschema in properties.items():
            if "default" in subschema:
                instance.setdefault(property, subschema["default"])

        for error in validate_properties(validator, properties, instance, schema):
            yield error

    return validators.extend(validator_class, {"properties" : set_defaults_and_validate})

DefaultValidatingDraft4Validator = extend_with_default(Draft4Validator)

def validate_and_inject_defaults(instance, schema, cls=None, *args, **kwargs):
    """
    Drop-in replacement for jsonschema.validate(), but also *modifies* the instance
    to fill missing properties with their schema-provided default values.

    See the jsonschema FAQ:
    http://python-jsonschema.readthedocs.org/en/latest/faq/
    """
    if cls is None:
        cls = validators.validator_for(schema)
    cls.check_schema(schema)

    # Add default-injection behavior to the validator
    extended_cls = extend_with_default(cls)
    
    # Validate and inject defaults.
    extended_cls(schema, *args, **kwargs).validate(instance)

if __name__ == "__main__":
    schema = {
        "type": "object",
        "properties": {
            "outer-object": {
                "type": "object",
                "properties" : {
                    "inner-object": {
                        "type": "string",
                        "default": "INNER-DEFAULT"
                    }
                },
                "default": {} # <-- MUST PROVIDE DEFAULT OBJECT
            }
        }
    }
    
    obj = {}
    DefaultValidatingDraft4Validator(schema).validate(obj)
    assert obj == {'outer-object': {'inner-object': 'INNER-DEFAULT'}}
    
    del schema["properties"]["outer-object"]["default"]    
    obj2 = {}
    DefaultValidatingDraft4Validator(schema).validate(obj2)
    assert obj2 == {} # whoops
    