from __future__ import print_function, absolute_import
from jsonschema import Draft4Validator, validators
import copy

def extend_with_default(validator_class):
    """
    This code was adapted from the jsonschema FAQ:
    http://python-jsonschema.readthedocs.org/en/latest/faq/
    """
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults_and_validate(validator, properties, instance, schema):
        for property, subschema in properties.items():
            if "default" in subschema:
                instance.setdefault(property, copy.copy(subschema["default"]))

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


def extend_with_default_without_validation(validator_class):
    """
    This code was adapted from the jsonschema FAQ:
    http://python-jsonschema.readthedocs.org/en/latest/faq/
    """
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for property, subschema in properties.items():
            if instance == "{{NO_DEFAULT}}":
                continue
            if "default" in subschema:
                instance.setdefault(property, copy.copy(subschema["default"]))
            else:
                instance.setdefault(property, "{{NO_DEFAULT}}")

        for _error in validate_properties(validator, properties, instance, schema):
            # Ignore validation errors
            pass

    def ignore_required(validator, required, instance, schema):
        return

    return validators.extend(validator_class, {"properties" : set_defaults, "required": ignore_required})

#DefaultInjector = extend_with_default(Draft4Validator)

def inject_defaults(instance, schema, cls=None, *args, **kwargs):
    """
    Modifies the given instance to inject defaults, where possible.
    Where no default is given, the string "{{NO_DEFAULT}}" is inserted
    instead (even if the property is not of type string).
    """
    if cls is None:
        cls = validators.validator_for(schema)
    cls.check_schema(schema)

    # Add default-injection behavior to the validator
    extended_cls = extend_with_default_without_validation(cls)
    
    # Inject defaults.
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
                    },
                    "inner-object-2": {
                        "type": "integer"
                    }
                },
                "default": {} # <-- MUST PROVIDE DEFAULT OBJECT
            }
        }
    }
    
    import json
    
    obj1 = {}
    DefaultValidatingDraft4Validator(schema).validate(obj1)
    assert obj1 == {'outer-object': {'inner-object': 'INNER-DEFAULT'}}

    print("obj1:")
    print(json.dumps(obj1, indent=4) + '\n')
        
    schema_nodefault = copy.deepcopy(schema)
    del schema_nodefault["properties"]["outer-object"]["default"]
    obj2 = {}
    DefaultValidatingDraft4Validator(schema_nodefault).validate(obj2)
    assert obj2 == {} # whoops

    print("obj2:")
    print(json.dumps(obj2, indent=4) + '\n')
    
    obj3 = {}
    inject_defaults(obj3, schema)

    print("obj3:")
    print(json.dumps(obj3, indent=4) + '\n')
