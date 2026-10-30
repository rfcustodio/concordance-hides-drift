"""Validate a Dynamic AI Safety Label against the JSON Schema.

Usage: python -I scripts/validate_label.py examples/dynamic_ai_safety_label.example.json
"""
import json
import os
import sys

import jsonschema

here = os.path.dirname(os.path.abspath(__file__))
schema = json.load(open(os.path.join(here, "..", "schema", "dynamic_ai_safety_label.schema.json")))
for path in sys.argv[1:]:
    doc = json.load(open(path))
    jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(doc)
    print(f"valid: {path}")
