from jsonformer.format import highlight_values
from jsonformer.main import Jsonformer
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading model and tokenizer...")
model_name = "databricks/dolly-v2-3b"
model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_cache=True)
print("Loaded model and tokenizer")

car = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "unit_cell": {
      "type": "object",
      "properties": {
        "a": { "type": "number" },
        "b": { "type": "number" },
        "c": { "type": "number" },
        "alpha": { "type": "number" },
        "beta": { "type": "number" },
        "gamma": { "type": "number" }
      },
      "required": ["a", "b", "c", "alpha", "beta", "gamma"]
    },
    "atoms": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "element": {
            "type": "string"
          },
          "atomic_number": {
            "type": "number"
          },
          "position": {
            "type": "object",
            "properties": {
              "x": {
                "type": "number"
              },
              "y": {
                "type": "number"
              },
              "z": {
                "type": "number"
              }
            },
            "required": ["x", "y", "z"]
          }
        },
        "required": ["element", "atomic_number", "position"]
      }
    },
  "target":{"type": "number"},
  },
  "required": ["unit_cell", "atoms","target"]
}

builder = Jsonformer(
    model=model,
    tokenizer=tokenizer,
    json_schema=car,
    prompt="Generate an example atomic structure",

)

print("Generating...")
output = builder()

highlight_values(output)
