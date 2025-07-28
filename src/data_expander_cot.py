"""
Data expander for Chain-of-Thought (CoT) examples.
Only includes examples that have _cot.txt files.
"""

import json
import random
from pathlib import Path
from typing import Dict, List
from xml.etree import ElementTree as ET
import xml.dom.minidom
from src.models.ocr_schemas import SchemaMapper
from src.prompt_generator import PromptGenerator


class CoTDataExpander:
    """Expand examples that have chain-of-thought traces."""
    
    # Same schema prompts as original expander
    SCHEMA1_XML_PROMPTS = [
        'Extract as XML with <article> root. Include: title(s), content sections with <p> paragraphs containing [N] markers, footnotes as <footnote n="N">, incomplete sections marked with incomplete="true"',
        'Convert to structured XML: <article> containing <title>, <section> with <p> elements. Mark [N] as footnote references, include <footnote n="N"> at end, flag partial sections with incomplete="true"'
    ]
    
    SCHEMA1_JSON_PROMPTS = [
        'Extract as JSON: {title: str, sections: [{content: [paragraphs with [N] markers], incomplete: bool}], footnotes: {N: text}}',
        'Convert to JSON with title, content sections (array of paragraphs), footnotes object {N: text}, incomplete flag for partial sections'
    ]
    
    SCHEMA2_XML_PROMPTS = [
        'Extract as XML with <page> root. Articles as <entry> with <heading>, body paragraphs. Footnotes inline as <note marker="N">',
        'Create XML: <page> with <entry> elements containing <heading> and paragraphs. Include footnotes inline using <note marker="N">'
    ]
    
    SCHEMA2_JSON_PROMPTS = [
        'Extract as JSON: {entries: [{heading: str, body: [text with {note: N, text: str} for footnotes]}]}',
        'Convert to JSON with entries array, each with heading and body paragraphs. Footnotes as {note: N, text: str} objects inline'
    ]
    
    SCHEMA3_XML_PROMPTS = [
        'Extract as XML: <content> with <header>, <section> elements. Use <footnote-ref n="N"/> for markers, <incomplete/> for partial text',
        'Create XML <content> with sections. Use [N] for annotation markers in text and incomplete="true" for partial sections.'
    ]
    
    SCHEMA3_JSON_PROMPTS = [
        'Extract as JSON: {header: str, sections: [text], footnotes: [{id: N, content: str}], flags: {incomplete: bool}}',
        'Convert to JSON with header, sections array, separate footnotes array with id/content, flags object for incomplete status'
    ]
    
    def __init__(self):
        self.prompt_generator = PromptGenerator()
        
    def expand_dataset(self) -> List[Dict]:
        """Main entry point to expand the dataset."""
        examples = self._load_examples()
        
        all_variations = []
        for example in examples:
            variations = self._generate_variations(example)
            all_variations.extend(variations)
            
        print(f"Generated {len(all_variations)} CoT variations from {len(examples)} examples")
        return all_variations
    
    def _load_examples(self) -> List[Dict]:
        """Load examples from ready directory - only those with CoT files."""
        examples = []
        ready_dir = Path("ready")
        
        for example_dir in ready_dir.iterdir():
            if not example_dir.is_dir():
                continue
                
            stem = example_dir.name
            
            # Check if CoT file exists - if not, skip this example
            cot_path = example_dir / f"{stem}_cot.txt"
            if not cot_path.exists():
                continue
                
            # Read CoT content
            with open(cot_path, 'r', encoding='utf-8') as f:
                cot_text = f.read().strip()
            
            # Find image file (with multiple possible extensions)
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                candidate = example_dir / f"{stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
                # Also check resized version
                candidate = example_dir / f"{stem}_resized{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            
            if not image_path:
                print(f"Warning: No image found for {stem} (has CoT)")
                continue
            
            # OCR is optional
            ocr_path = example_dir / f"{stem}_ocr.txt"
            ocr_text = ""
            if ocr_path.exists():
                with open(ocr_path, 'r', encoding='utf-8') as f:
                    ocr_text = f.read().strip()
            
            # Find output file (XML or JSON)
            xml_path = example_dir / f"{stem}_done.xml"
            json_path = example_dir / f"{stem}_done.json"
            
            xml_output = None
            json_output = None
            
            if xml_path.exists():
                with open(xml_path, 'r', encoding='utf-8') as f:
                    xml_output = f.read().strip()
            
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_output = f.read().strip()
            
            if not xml_output and not json_output:
                print(f"Warning: No output file found for {stem}")
                continue
            
            # Read prompt if exists
            prompt_path = example_dir / f"{stem}_prompt.txt"
            prompt = None
            if prompt_path.exists():
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    prompt = f.read().strip()
            
            examples.append({
                "id": stem,
                "image_path": str(image_path),
                "ocr_text": ocr_text,
                "xml_output": xml_output,
                "json_output": json_output,
                "cot_text": cot_text,
                "original_prompt": prompt
            })
        
        return examples
    
    def _add_think_to_prompt(self, prompt: str, seed: int) -> str:
        """Randomly add 'think' to the beginning or end of prompt."""
        random.seed(seed)
        position = random.choice(['start', 'end'])
        
        if position == 'start':
            return f"think\n{prompt}"
        else:
            return f"{prompt}\nthink"
    
    def _format_cot_output(self, cot_text: str, output: str) -> str:
        """Format output with CoT trace in <thk> tags."""
        return f"<thk>\n{cot_text}\n</thk>\n{output}"
    
    def _generate_variations(self, example: Dict) -> List[Dict]:
        """Generate all variations for a single example."""
        variations = []
        
        # Only process known example types (allowlist approach)
        if "enc_brit" in example["id"]:
            # Encyclopedia examples get schema variations
            variations.extend(self._generate_encyclopedia_variations(example))
        elif "hackernews" in example["id"]:
            # Hackernews examples get simple variations
            variations.extend(self._generate_simple_variations(example))
        else:
            # For all other examples, include them as-is
            variations.extend(self._generate_generic_variations(example))
            
        return variations
    
    def _generate_encyclopedia_variations(self, example: Dict) -> List[Dict]:
        """Generate schema variations for encyclopedia examples."""
        variations = []
        
        # Parse original XML into Schema1
        try:
            doc1 = self._parse_original_xml(example["xml_output"])
        except Exception as e:
            print(f"Error parsing {example['id']}: {e}")
            return variations
        
        # Generate variations for each schema
        schemas = [
            ("schema1", doc1, self.SCHEMA1_XML_PROMPTS, self.SCHEMA1_JSON_PROMPTS),
            ("schema2", SchemaMapper.schema1_to_schema2(doc1), self.SCHEMA2_XML_PROMPTS, self.SCHEMA2_JSON_PROMPTS),
            ("schema3", SchemaMapper.schema1_to_schema3(doc1), self.SCHEMA3_XML_PROMPTS, self.SCHEMA3_JSON_PROMPTS),
        ]
        
        for schema_name, doc, xml_prompts, json_prompts in schemas:
            # XML variations
            xml_str = self._doc_to_xml_string(doc)
            for i, prompt in enumerate(xml_prompts):
                # Add think to prompt
                cot_prompt = self._add_think_to_prompt(prompt, hash(f"{example['id']}_{schema_name}_xml_{i}"))
                # Format output with CoT
                cot_output = self._format_cot_output(example['cot_text'], xml_str)
                
                variations.append({
                    "id": f"{example['id']}_{schema_name}_xml_cot_p{i}",
                    "image_path": example["image_path"],
                    "ocr_text": example["ocr_text"],
                    "prompt": cot_prompt,
                    "output": cot_output,
                    "format": "xml",
                    "schema": schema_name
                })
            
            # JSON variations
            json_obj = SchemaMapper.xml_to_json(doc, schema_name)
            json_str = json.dumps(json_obj, indent=2, ensure_ascii=False)
            for i, prompt in enumerate(json_prompts):
                # Add think to prompt
                cot_prompt = self._add_think_to_prompt(prompt, hash(f"{example['id']}_{schema_name}_json_{i}"))
                # Format output with CoT
                cot_output = self._format_cot_output(example['cot_text'], json_str)
                
                variations.append({
                    "id": f"{example['id']}_{schema_name}_json_cot_p{i}",
                    "image_path": example["image_path"],
                    "ocr_text": example["ocr_text"],
                    "prompt": cot_prompt,
                    "output": cot_output,
                    "format": "json",
                    "schema": schema_name
                })
        
        return variations
    
    def _generate_simple_variations(self, example: Dict) -> List[Dict]:
        """Generate simple variations for hackernews examples."""
        variations = []
        
        # Hardcoded prompts for hackernews
        json_prompts = [
            'Extract as JSON: {stories: [{rank: N, title: "...", link: "...", points: N, user: "...", time: "...", comments: N}]}',
            'Extract as valid JSON: {stories: [{rank: N, title: "...", link: "...", points: N, user: "...", time: "...", comments: N}]}'
        ]
        xml_prompts = [
            'Extract as XML: <stories> with <story rank="N"> containing <title>, <link>, <points>, <user>, <time>, <comments>',
            'Convert to XML: <stories> containing <story rank="N"> with child elements for title, link, points, user, time, and comments'
        ]
        
        # If we have JSON, create both JSON and XML variations
        if example["json_output"]:
            # JSON variations
            for i, prompt in enumerate(json_prompts):
                cot_prompt = self._add_think_to_prompt(prompt, hash(f"{example['id']}_json_{i}"))
                cot_output = self._format_cot_output(example['cot_text'], example["json_output"])
                
                variations.append({
                    "id": f"{example['id']}_json_cot_p{i}",
                    "image_path": example["image_path"],
                    "ocr_text": example.get("ocr_text", ""),
                    "prompt": cot_prompt,
                    "output": cot_output,
                    "format": "json",
                    "schema": "hackernews"
                })
            
            # Create XML from JSON for XML variations
            xml_output = self._json_to_simple_xml(example["json_output"], example["id"])
            for i, prompt in enumerate(xml_prompts):
                cot_prompt = self._add_think_to_prompt(prompt, hash(f"{example['id']}_xml_{i}"))
                cot_output = self._format_cot_output(example['cot_text'], xml_output)
                
                variations.append({
                    "id": f"{example['id']}_xml_cot_p{i}",
                    "image_path": example["image_path"],
                    "ocr_text": example.get("ocr_text", ""),
                    "prompt": cot_prompt,
                    "output": cot_output,
                    "format": "xml",
                    "schema": "hackernews"
                })
        
        return variations
    
    def _generate_generic_variations(self, example: Dict) -> List[Dict]:
        """Generate variations for other examples using their original prompts."""
        variations = []
        
        # Use original prompt or default
        base_prompt = example.get('original_prompt', 'Extract as structured data')
        
        # Determine format based on which output file exists
        if example.get("xml_output"):
            cot_prompt = self._add_think_to_prompt(base_prompt, hash(f"{example['id']}_xml"))
            cot_output = self._format_cot_output(example['cot_text'], example["xml_output"])
            
            variations.append({
                "id": f"{example['id']}_cot_original",
                "image_path": example["image_path"],
                "ocr_text": example.get("ocr_text", ""),
                "prompt": cot_prompt,
                "output": cot_output,
                "format": "xml",
                "schema": "original"
            })
        elif example.get("json_output"):
            cot_prompt = self._add_think_to_prompt(base_prompt, hash(f"{example['id']}_json"))
            cot_output = self._format_cot_output(example['cot_text'], example["json_output"])
            
            variations.append({
                "id": f"{example['id']}_cot_original",
                "image_path": example["image_path"],
                "ocr_text": example.get("ocr_text", ""),
                "prompt": cot_prompt,
                "output": cot_output,
                "format": "json",
                "schema": "original"
            })
        
        return variations
    
    def _parse_original_xml(self, xml_str: str):
        """Parse original XML into Schema1Document."""
        # Implementation copied from original data_expander.py
        from src.models.ocr_schemas import Schema1Document, Article, Section, Footnote
        
        root = ET.fromstring(xml_str)
        
        articles = []
        for article_elem in root.findall(".//article"):
            title_elem = article_elem.find("title")
            title = title_elem.text if title_elem is not None else ""
            
            sections = []
            for section_elem in article_elem.findall("section"):
                content = []
                for p_elem in section_elem.findall("p"):
                    if p_elem.text:
                        content.append(p_elem.text.strip())
                
                incomplete = section_elem.get("incomplete") == "true"
                sections.append(Section(content=content, incomplete=incomplete))
            
            footnotes = []
            for footnote_elem in article_elem.findall("footnote"):
                n = footnote_elem.get("n", "")
                text = footnote_elem.text.strip() if footnote_elem.text else ""
                footnotes.append(Footnote(n=n, text=text))
            
            articles.append(Article(
                title=title,
                sections=sections,
                footnotes=footnotes
            ))
        
        return Schema1Document(articles=articles)
    
    def _doc_to_xml_string(self, doc) -> str:
        """Convert document to formatted XML string."""
        xml_str = doc.to_xml(skip_empty=True, encoding=None)
        return self._prettify_xml(xml_str)
    
    def _prettify_xml(self, xml_str: str) -> str:
        """Format XML with proper indentation."""
        dom = xml.dom.minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent="    ", encoding=None)
        lines = [line for line in pretty_xml.split('\n') if line.strip()]
        if lines and lines[0].startswith('<?xml'):
            lines = lines[1:]
        return '\n'.join(lines)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text by collapsing whitespace."""
        return ' '.join(text.split())
    
    def _json_to_simple_xml(self, json_str: str, example_id: str) -> str:
        """Convert JSON to simple XML for non-encyclopedia examples."""
        try:
            data = json.loads(json_str)
            
            if "stories" in data:
                # Hackernews format
                xml_lines = ['<stories>']
                for story in data.get("stories", []):
                    rank = story.get("rank", "")
                    xml_lines.append(f'    <story rank="{rank}">')
                    xml_lines.append(f'        <title>{story.get("title", "")}</title>')
                    xml_lines.append(f'        <link>{story.get("link", "")}</link>')
                    xml_lines.append(f'        <points>{story.get("points", "")}</points>')
                    xml_lines.append(f'        <user>{story.get("user", "")}</user>')
                    xml_lines.append(f'        <time>{story.get("time", "")}</time>')
                    xml_lines.append(f'        <comments>{story.get("comments", "")}</comments>')
                    xml_lines.append('    </story>')
                xml_lines.append('</stories>')
                return '\n'.join(xml_lines)
            else:
                # Generic conversion
                return f"<data>{json_str}</data>"
                
        except Exception as e:
            print(f"Error converting JSON to XML for {example_id}: {e}")
            return f"<data>{json_str}</data>"
    
    def save_to_file(self, variations: List[Dict], output_path: str):
        """Save expanded dataset to JSON file."""
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(variations, f, indent=2, ensure_ascii=False)
        
        # Also save manifest
        manifest = {
            "total_variations": len(variations),
            "cot_examples_only": True,
            "by_format": {
                "xml": len([v for v in variations if v["format"] == "xml"]),
                "json": len([v for v in variations if v["format"] == "json"])
            },
            "by_schema": {}
        }
        
        for schema in set(v["schema"] for v in variations):
            manifest["by_schema"][schema] = len([v for v in variations if v["schema"] == schema])
        
        manifest_path = output_dir / "cot_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)


def main():
    """Main entry point."""
    expander = CoTDataExpander()
    variations = expander.expand_dataset()
    expander.save_to_file(variations, "outputs-synth/cot_expanded_dataset.json")
    print(f"Saved to outputs-synth/cot_expanded_dataset.json")


if __name__ == "__main__":
    main()