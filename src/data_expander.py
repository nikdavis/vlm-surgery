"""
Data expander that generates schema variations and formats for training data.
Takes examples from ready/ and outputs to outputs-synth/.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import xml.etree.ElementTree as ET

from src.models.ocr_schemas import (
    Schema1Document, Schema1Article, Schema1Footnote,
    Schema2Document, Schema3Document, SchemaMapper
)


class DataExpander:
    """Expands OCR training data with schema and format variations."""
    
    # Prompts for different schemas and formats
    SCHEMA1_XML_PROMPTS = [
        "Extract as XML within <document> root: <header>, <article continuation=\"true\"?> containing <title>?, <body> with [N] marking footnote references, <footnotes><footnote id=\"N\"> for corresponding notes, <bibliography>?",
        "Convert to XML <document> structure with header and articles. Mark continued articles with continuation=\"true\". Use [N] for footnote references in body text.",
    ]
    
    SCHEMA1_JSON_PROMPTS = [
        "Extract as JSON with header and articles array. In body text, mark footnote references as [N]. Include footnotes array with {id: N, text: \"...\"}. Mark continued articles with continuation: true.",
        "Convert to JSON: {header: \"...\", articles: [{title?: \"...\", body: \"text with [N] for footnotes\", footnotes?: [{id: N, text: \"...\"}], bibliography?: \"...\", continuation?: true}]}",
    ]
    
    SCHEMA2_XML_PROMPTS = [
        "Parse into XML <page>: <header>, <entry continued=\"true\"?> with <heading>?, <content> containing [N] for note references, <notes><note id=\"N\">, <references>?",
        "Transform to XML <page> using entry/content structure. Use [N] for note markers in content and continued=\"true\" for incomplete entries.",
    ]
    
    SCHEMA2_JSON_PROMPTS = [
        "Parse as JSON: {header: \"...\", entries: [{heading?: \"...\", content: \"text with [N] markers\", notes?: [{id: N, text: \"...\"}], references?: \"...\", continued?: true}]}",
        "Generate JSON with header and entries. Mark note references as [N] in content. Include notes array and continued flag where applicable.",
    ]
    
    SCHEMA3_XML_PROMPTS = [
        "Structure as XML <content>: <header>, <section incomplete=\"true\"?> containing <caption>?, <text> with [N] for annotations, <annotations><annotation id=\"N\">, <sources>?",
        "Create XML <content> with sections. Use [N] for annotation markers in text and incomplete=\"true\" for partial sections.",
    ]
    
    SCHEMA3_JSON_PROMPTS = [
        "Format as JSON: {header: \"...\", sections: [{caption?: \"...\", text: \"content with [N] marks\", annotations?: [{id: N, text: \"...\"}], sources?: \"...\", incomplete?: true}]}",
        "Build JSON structure with header and sections. Use [N] for annotation references. Include incomplete flag for partial sections.",
    ]
    
    def __init__(
        self,
        ready_dir: Path = Path("ready"),
        output_dir: Path = Path("outputs-synth"),
        use_llm_for_prompts: bool = False
    ):
        self.ready_dir = ready_dir
        self.output_dir = output_dir
        self.use_llm_for_prompts = use_llm_for_prompts
        self.output_dir.mkdir(exist_ok=True)
        
        # Load generated prompts if available
        self.generated_prompts = {}
        prompts_file = Path("prompts/generated_prompts.json")
        if prompts_file.exists():
            with open(prompts_file, 'r') as f:
                self.generated_prompts = json.load(f)
        
    def expand_all(self):
        """Process all examples and generate variations."""
        examples = self._load_examples()
        print(f"Loaded {len(examples)} examples from {self.ready_dir}")
        
        all_variations = []
        
        for example in examples:
            variations = self._generate_variations(example)
            all_variations.extend(variations)
            
        # Save all variations
        output_file = self.output_dir / "expanded_dataset.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_variations, f, indent=2, ensure_ascii=False)
            
        print(f"Generated {len(all_variations)} variations")
        print(f"Saved to {output_file}")
        
        # Also save a manifest for easy loading
        manifest = {
            "total_examples": len(all_variations),
            "original_examples": len(examples),
            "schemas": ["schema1", "schema2", "schema3"],
            "formats": ["xml", "json"],
            "prompts_per_combo": 2
        }
        
        manifest_file = self.output_dir / "manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def _load_examples(self) -> List[Dict]:
        """Load all examples from ready directory."""
        examples = []
        
        for example_dir in sorted(self.ready_dir.iterdir()):
            if not example_dir.is_dir():
                continue
            
            # Find image file (support jpg, jpeg, png)
            stem = example_dir.name
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png']:
                candidate = example_dir / f"{stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            
            if not image_path:
                print(f"Warning: No image found for {stem}")
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
            
            examples.append({
                "id": stem,
                "image_path": str(image_path),
                "ocr_text": ocr_text,
                "xml_output": xml_output,
                "json_output": json_output
            })
        
        return examples
    
    def _generate_variations(self, example: Dict) -> List[Dict]:
        """Generate all variations for a single example."""
        variations = []
        
        # Only process known example types (allowlist approach)
        if "enc_brit" in example["id"]:
            # Encyclopedia examples get schema variations
            pass  # Continue to schema processing below
        elif "hackernews" in example["id"]:
            # Hackernews examples get simple variations
            return self._generate_simple_variations(example)
        else:
            # For all other examples, include them as-is without expansion
            # Read the actual prompt file if it exists
            stem = example['id']
            prompt_path = Path(f"ready/{stem}/{stem}_prompt.txt")
            prompt = "Extract as structured data"
            
            if prompt_path.exists():
                with open(prompt_path, 'r') as f:
                    prompt = f.read().strip()
            
            # Determine format based on which output file exists
            if example.get("xml_output"):
                return [{
                    "id": f"{example['id']}_original",
                    "image_path": example["image_path"],
                    "ocr_text": example.get("ocr_text", ""),
                    "prompt": prompt,
                    "output": example["xml_output"],
                    "format": "xml",
                    "schema": "original"
                }]
            elif example.get("json_output"):
                return [{
                    "id": f"{example['id']}_original",
                    "image_path": example["image_path"],
                    "ocr_text": example.get("ocr_text", ""),
                    "prompt": prompt,
                    "output": example["json_output"],
                    "format": "json",
                    "schema": "original"
                }]
            else:
                return []
        
        # Parse original XML into Schema1 for encyclopedia examples
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
            # No processing needed - we keep [N] format for all schemas
            
            # XML variations
            xml_output = doc.to_xml()
            # Ensure it's a string, not bytes
            if isinstance(xml_output, bytes):
                xml_output = xml_output.decode('utf-8')
            for i, prompt in enumerate(xml_prompts):
                variations.append({
                    "id": f"{example['id']}_{schema_name}_xml_p{i}",
                    "image_path": example["image_path"],
                    "ocr_text": example["ocr_text"],
                    "prompt": prompt,
                    "output": xml_output,
                    "format": "xml",
                    "schema": schema_name
                })
            
            # JSON variations
            json_dict = json.loads(doc.model_dump_json(exclude_none=True))
            json_output = json.dumps(json_dict, indent=2, ensure_ascii=False)
            for i, prompt in enumerate(json_prompts):
                variations.append({
                    "id": f"{example['id']}_{schema_name}_json_p{i}",
                    "image_path": example["image_path"],
                    "ocr_text": example["ocr_text"],
                    "prompt": prompt,
                    "output": json_output,
                    "format": "json",
                    "schema": schema_name
                })
        
        return variations
    
    def _parse_original_xml(self, xml_string: str) -> Schema1Document:
        """Parse original XML format into Schema1."""
        # Clean up the XML string - escape common problematic characters
        # But be careful not to double-escape
        if '&amp;' not in xml_string and '&lt;' not in xml_string:
            xml_string = xml_string.replace('&', '&amp;')
        
        # Wrap if needed
        if not xml_string.strip().startswith('<?xml'):
            xml_string = f"<root>{xml_string}</root>"
        
        try:
            root = ET.fromstring(xml_string)
        except ET.ParseError as e:
            # Try to fix common issues
            print(f"Initial parse error: {e}")
            raise
        
        # If we wrapped it, get the actual elements
        if root.tag == 'root':
            elements = list(root)
        else:
            elements = [root]
        
        header = ""
        articles = []
        
        for elem in elements:
            if elem.tag == 'header':
                header = elem.text.strip() if elem.text else ""
            elif elem.tag == 'article':
                article = self._parse_article(elem)
                articles.append(article)
        
        return Schema1Document(header=header, articles=articles)
    
    def _parse_article(self, article_elem: ET.Element) -> Schema1Article:
        """Parse an article element."""
        continuation = 'true' if article_elem.get('continuation') == 'true' else None
        
        title = None
        title_elem = article_elem.find('title')
        if title_elem is not None and title_elem.text:
            title = title_elem.text.strip()
        
        body = ""
        body_elem = article_elem.find('body')
        if body_elem is not None:
            # Extract text with [N] markers instead of XML refs
            body = self._extract_body_text(body_elem)
        
        footnotes = []
        footnotes_elem = article_elem.find('footnotes')
        if footnotes_elem is not None:
            for footnote in footnotes_elem.findall('footnote'):
                fn_id = footnote.get('id')
                fn_text = footnote.text
                if fn_id and fn_text:
                    footnotes.append(Schema1Footnote(
                        id=int(fn_id),
                        text=fn_text.strip()
                    ))
        
        bibliography = None
        biblio_elem = article_elem.find('bibliography')
        if biblio_elem is not None and biblio_elem.text:
            bibliography = biblio_elem.text.strip()
        
        return Schema1Article(
            continuation=continuation,
            title=title,
            body=body,
            footnotes=footnotes if footnotes else None,
            bibliography=bibliography
        )
    
    def _extract_body_text(self, elem: ET.Element) -> str:
        """Extract text from body element, converting refs to [N] markers."""
        parts = []
        
        if elem.text:
            parts.append(elem.text)
        
        for child in elem:
            if child.tag == 'ref':
                ref_id = child.get('id', '?')
                parts.append(f'[{ref_id}]')
            else:
                parts.append(self._extract_body_text(child))
            if child.tail:
                parts.append(child.tail)
        
        # Join and normalize whitespace
        text = ' '.join(parts)
        return ' '.join(text.split())
    
    def _generate_simple_variations(self, example: Dict) -> List[Dict]:
        """Generate simple variations for non-encyclopedia examples (e.g., hackernews)."""
        variations = []
        
        # Check if we have generated prompts for this example
        if example["id"] in self.generated_prompts:
            prompts = self.generated_prompts[example["id"]]
            json_prompts = prompts.get("json_prompts", ["Extract as JSON", "Convert to JSON format"])
            xml_prompts = prompts.get("xml_prompts", ["Extract as XML", "Convert to XML format"])
        elif "hackernews" in example["id"]:
            # Hardcoded prompts for hackernews
            json_prompts = [
                'Extract as JSON: {stories: [{rank: N, title: "...", link: "...", points: N, user: "...", time: "...", comments: N}]}',
                'Extract as valid JSON: {stories: [{rank: N, title: "...", link: "...", points: N, user: "...", time: "...", comments: N}]}'
            ]
            xml_prompts = [
                'Extract as XML: <stories> with <story rank="N"> containing <title>, <link>, <points>, <user>, <time>, <comments>',
                'Convert to XML: <stories> containing <story rank="N"> with child elements for title, link, points, user, time, and comments'
            ]
        elif "pge" in example["id"]:
            # Check if there's a prompt file
            prompt_file = Path(f"ready/{example['id']}/{example['id']}_prompt.txt")
            if prompt_file.exists():
                with open(prompt_file, 'r') as f:
                    base_prompt = f.read().strip()
                # Use the prompt for both XML variations
                xml_prompts = [base_prompt, base_prompt]
                # For JSON, adapt the prompt
                json_prompts = [
                    base_prompt.replace('as XML:', 'as JSON:').replace('</page>', '').replace('<page>', ''),
                    'Extract as JSON with same structure: account_info with fields, charges with groups containing name/total/items'
                ]
            else:
                # Fallback prompts for PGE
                xml_prompts = [
                    'Extract as XML: <page> with <account_info> containing <field name="..." value="..."/>, <charges> with <group> elements having <name>, <total>, and <item description="..." amount="..."/>',
                    'Convert to XML structure with account info fields and charge groups'
                ]
                json_prompts = ["Extract as JSON", "Convert to JSON format"]
        else:
            # Generic prompts for other types
            json_prompts = ["Extract as JSON", "Convert to JSON format"]
            xml_prompts = ["Extract as XML", "Convert to XML format"]
        
        # If we have JSON, create both JSON and XML variations
        if example["json_output"]:
            # JSON variations
            for i, prompt in enumerate(json_prompts):
                variations.append({
                    "id": f"{example['id']}_json_p{i}",
                    "image_path": example["image_path"],
                    "ocr_text": example["ocr_text"],
                    "prompt": prompt,
                    "output": example["json_output"],
                    "format": "json",
                    "schema": "original"
                })
            
            # Convert JSON to XML if possible
            try:
                xml_output = self._json_to_simple_xml(example["json_output"], example["id"])
                for i, prompt in enumerate(xml_prompts):
                    variations.append({
                        "id": f"{example['id']}_xml_p{i}",
                        "image_path": example["image_path"],
                        "ocr_text": example["ocr_text"],
                        "prompt": prompt,
                        "output": xml_output,
                        "format": "xml",
                        "schema": "original"
                    })
            except Exception as e:
                print(f"Could not convert JSON to XML for {example['id']}: {e}")
        
        # If we have XML, create variations (shouldn't happen for hackernews but just in case)
        elif example["xml_output"]:
            # XML variations
            for i, prompt in enumerate(xml_prompts):
                variations.append({
                    "id": f"{example['id']}_xml_p{i}",
                    "image_path": example["image_path"],
                    "ocr_text": example["ocr_text"],
                    "prompt": prompt,
                    "output": example["xml_output"],
                    "format": "xml",
                    "schema": "original"
                })
        
        return variations
    
    def _json_to_simple_xml(self, json_str: str, example_id: str) -> str:
        """Convert JSON to XML for simple schemas like hackernews."""
        data = json.loads(json_str)
        
        if "hackernews" in example_id and "stories" in data:
            # Convert hackernews format
            root = ET.Element("stories")
            for story in data["stories"]:
                story_elem = ET.SubElement(root, "story", rank=str(story.get("rank", "")))
                for key, value in story.items():
                    if key != "rank":
                        child = ET.SubElement(story_elem, key)
                        child.text = str(value)
            return self._prettify_xml(root)
        else:
            # Generic conversion
            root = ET.Element("root")
            self._dict_to_xml(data, root)
            return self._prettify_xml(root)
    
    def _dict_to_xml(self, data: Any, parent: ET.Element):
        """Recursively convert dict/list to XML."""
        if isinstance(data, dict):
            for key, value in data.items():
                child = ET.SubElement(parent, key)
                self._dict_to_xml(value, child)
        elif isinstance(data, list):
            for item in data:
                child = ET.SubElement(parent, "item")
                self._dict_to_xml(item, child)
        else:
            parent.text = str(data)
    
    def _prettify_xml(self, elem: ET.Element) -> str:
        """Return a pretty-printed XML string."""
        from xml.dom import minidom
        
        rough_string = ET.tostring(elem, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        # Remove the XML declaration
        return '\n'.join(reparsed.toprettyxml(indent="  ").split('\n')[1:])


if __name__ == "__main__":
    expander = DataExpander()
    expander.expand_all()