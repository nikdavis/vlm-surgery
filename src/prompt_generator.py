#!/usr/bin/env python3
"""
Generate prompts for data extraction using LLM analysis of the output structure.
"""

import os
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from pathlib import Path
from pydantic import BaseModel
from litellm import completion

MODEL="gemini/gemini-2.5-pro"
TEMPERATURE=0.4
MAX_TOKENS=8192

class ExtractedSchema(BaseModel):
    """Schema information extracted from example data."""
    root_element: str
    main_container: str
    item_element: str
    attributes: List[str]
    child_elements: List[str]
    has_nested_structure: bool

class PromptVariations(BaseModel):
    """Generated prompt variations."""
    xml_prompts: List[str]
    json_prompts: List[str]


class PromptGenerator:
    """Generate prompts by analyzing example data structures."""

    def __init__(self):
        self.model = MODEL
        self.temperature = TEMPERATURE
        self.max_tokens = MAX_TOKENS

    def analyze_json_structure(self, json_data: str) -> Dict:
        """Analyze JSON structure to understand the schema."""
        data = json.loads(json_data)

        # Build a description of the structure
        structure_desc = self._describe_structure(data)

        # Use LLM to analyze and suggest prompts
        messages = [{
            "role": "system",
            "content": """You are a data structure analyzer. Analyze the JSON structure and generate extraction prompts.
            Focus on the data hierarchy, field names, and data types."""
        }, {
            "role": "user",
            "content": f"""Analyze this JSON structure and generate prompts for extracting similar data:

{json.dumps(data, indent=2)[:1000]}...

Provide:
1. Two XML extraction prompts that would result in a similar structure
2. Two JSON extraction prompts with slight variations
3. Keep prompts concise and specific to the data fields shown"""
        }]

        response = completion(
            model=self.model,
            messages=messages,
            response_format=PromptVariations,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return response.choices[0].message.content

    def analyze_xml_structure(self, xml_data: str) -> Dict:
        """Analyze XML structure to understand the schema."""
        try:
            root = ET.fromstring(xml_data)
        except:
            # Try wrapping in root
            root = ET.fromstring(f"<root>{xml_data}</root>")

        # Use LLM to analyze
        messages = [{
            "role": "system",
            "content": """You are a data structure analyzer. Analyze the XML structure and generate extraction prompts.
            Focus on element names, attributes, and hierarchy."""
        }, {
            "role": "user",
            "content": f"""Analyze this XML structure and generate prompts for extracting similar data:

{xml_data[:1000]}...

Provide:
1. Two XML extraction prompts that match this exact structure
2. Two JSON extraction prompts that would extract the same information
3. Keep prompts concise and mention key elements/attributes"""
        }]

        response = completion(
            model=self.model,
            messages=messages,
            response_format=PromptVariations,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return response.choices[0].message.content

    def _describe_structure(self, data, level=0, max_level=3):
        """Recursively describe data structure."""
        if level > max_level:
            return "..."

        if isinstance(data, dict):
            items = []
            for k, v in list(data.items())[:5]:  # Limit to first 5 items
                items.append(f"{k}: {self._describe_structure(v, level+1, max_level)}")
            return "{" + ", ".join(items) + "}"
        elif isinstance(data, list):
            if data:
                return f"[{self._describe_structure(data[0], level+1, max_level)}, ...]"
            return "[]"
        else:
            return type(data).__name__

    def generate_prompts_for_example(self, example_path: Path) -> Dict[str, List[str]]:
        """Generate prompts for a specific example."""
        stem = example_path.name

        # Check what files we have
        json_path = example_path / f"{stem}_done.json"
        xml_path = example_path / f"{stem}_done.xml"

        prompts = {
            "xml_prompts": [],
            "json_prompts": []
        }

        try:
            if json_path.exists():
                with open(json_path, 'r') as f:
                    json_data = f.read()
                result = self.analyze_json_structure(json_data)
                if isinstance(result, str):
                    result = json.loads(result)
                prompts["xml_prompts"].extend(result.get("xml_prompts", []))
                prompts["json_prompts"].extend(result.get("json_prompts", []))

            elif xml_path.exists():
                with open(xml_path, 'r') as f:
                    xml_data = f.read()
                result = self.analyze_xml_structure(xml_data)
                if isinstance(result, str):
                    result = json.loads(result)
                prompts["xml_prompts"].extend(result.get("xml_prompts", []))
                prompts["json_prompts"].extend(result.get("json_prompts", []))

        except Exception as e:
            print(f"Error generating prompts for {stem}: {e}")

        return prompts

    def update_expander_prompts(self, ready_dir: Path = Path("ready")):
        """Generate and save prompts for all non-standard examples."""
        all_prompts = {}

        for example_dir in ready_dir.iterdir():
            if not example_dir.is_dir():
                continue

            # Skip encyclopedia examples (they have hardcoded prompts)
            if example_dir.name.startswith("enc_brit"):
                continue

            print(f"Generating prompts for {example_dir.name}...")
            prompts = self.generate_prompts_for_example(example_dir)

            if prompts["xml_prompts"] or prompts["json_prompts"]:
                all_prompts[example_dir.name] = prompts

        # Save generated prompts
        output_file = Path("prompts/generated_prompts.json")
        output_file.parent.mkdir(exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(all_prompts, f, indent=2)

        print(f"Saved prompts to {output_file}")
        return all_prompts


def main():
    """Generate prompts for new examples."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate prompts using LLM")
    parser.add_argument("--example", type=Path, help="Generate for specific example")
    parser.add_argument("--all", action="store_true", help="Generate for all non-encyclopedia examples")

    args = parser.parse_args()

    generator = PromptGenerator()

    if args.example:
        prompts = generator.generate_prompts_for_example(args.example)
        print(json.dumps(prompts, indent=2))
    elif args.all:
        generator.update_expander_prompts()
    else:
        # Test with hackernews
        test_json = '''{"stories": [{"rank": 1, "title": "Example", "link": "example.com", "points": 100, "user": "test", "time": "1 hour ago", "comments": 50}]}'''
        result = generator.analyze_json_structure(test_json)
        print(result)


if __name__ == "__main__":
    main()
