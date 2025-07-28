"""
Dataset loader for OCR fine-tuning examples.
Loads approved examples from ./ready/ folder.
"""

from pathlib import Path
from typing import List, Dict, Optional, Any
from PIL import Image
import random
import json
import xml.etree.ElementTree as ET


class OCRDataset:
    """Load OCR examples for fine-tuning."""

    # XML format prompts
    XML_PROMPTS = [
        "Extract as XML within <document> root: <header>, <article continuation=\"true\"?> containing <title>?, <body> with <ref id=\"N\"/> marking footnote references, <footnotes><footnote id=\"N\"> for corresponding notes, <bibliography>?",
        "Convert to XML <document>: header, then articles (each has: optional title, required body with inline <ref id=\"N\"/> markers linking to <footnotes><footnote id=\"N\">, optional bibliography). Mark continued articles with continuation=\"true\".",
    ]
    
    # JSON format prompts
    JSON_PROMPTS = [
        "Extract as JSON with header and articles array. In body text, mark footnote references as [N]. Include footnotes array with {id: N, text: \"...\"}. Mark continued articles with continuation: true.",
        "Convert to JSON: {header: \"...\", articles: [{title?: \"...\", body: \"text with [N] for footnotes\", footnotes?: [{id: N, text: \"...\"}], bibliography?: \"...\", continuation?: true}]}",
    ]
    
    # Committee hearing format prompts (for future use)
    COMMITTEE_XML_PROMPTS = [
        "Extract hearing transcript as XML: <page><title>, <number>, <testimonies> with <testimony><speaker>, <text>, <type>question|answer|statement</type></testimony>",
        "Convert to XML page structure with title, number, and testimony elements. Each testimony has speaker name, text content, and type classification."
    ]
    
    COMMITTEE_JSON_PROMPTS = [
        "Extract as JSON: {title: \"...\", number: N, testimonies: [{speaker: \"...\", text: \"...\", type: \"question|answer|statement\"}]}",
        "Convert hearing to JSON with page title, number, and testimonies array containing speaker, text, and type for each exchange."
    ]

    def __init__(self, ready_dir: Path = Path("ready"), randomize_prompts: bool = False, augment_formats: bool = True, augment_prompts: bool = True):
        self.ready_dir = ready_dir
        self.randomize_prompts = randomize_prompts  # If True, randomly select prompts. If False, systematically iterate through all.
        self.augment_formats = augment_formats  # Double dataset with XML and JSON
        self.augment_prompts = augment_prompts  # Double dataset again with prompt variations
        self.examples = self._load_examples()

    def _load_examples(self) -> List[Dict]:
        """Load all examples from ready directory."""
        examples = []

        # Find all example directories
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
            
            if xml_path.exists():
                with open(xml_path, 'r', encoding='utf-8') as f:
                    xml_output = f.read().strip()
                json_output = None
            elif json_path.exists():
                # If only JSON exists, convert it to XML later
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_output = f.read().strip()
                xml_output = None
            else:
                print(f"Warning: No output file found for {stem}")
                continue

            # Create example
            example = {
                "id": stem,
                "image_path": str(image_path),
                "ocr_text": ocr_text,
                "xml_output": xml_output,
                "json_output": json_output
            }

            examples.append(example)

        print(f"Loaded {len(examples)} examples from {self.ready_dir}")
        return examples

    def _parse_article(self, article_elem: ET.Element) -> Dict[str, Any]:
        """Parse an article element into a dictionary."""
        article_data = {}
        
        # Handle continuation attribute
        if article_elem.get('continuation') == 'true':
            article_data['continuation'] = True
        
        # Extract title (optional)
        title_elem = article_elem.find('title')
        if title_elem is not None and title_elem.text:
            article_data['title'] = title_elem.text.strip()
        
        # Extract body with ref markers
        body_elem = article_elem.find('body')
        if body_elem is not None:
            # Get all text content recursively, preserving ref markers
            def get_all_text(elem):
                parts = []
                if elem.text:
                    parts.append(elem.text)
                for child in elem:
                    if child.tag == 'ref':
                        ref_id = child.get('id', '?')
                        parts.append(f'[{ref_id}]')
                    else:
                        parts.extend(get_all_text(child))
                    if child.tail:
                        parts.append(child.tail)
                return parts
            
            text_parts = get_all_text(body_elem)
            body_text = ' '.join(text_parts)
            # Normalize whitespace
            body_text = ' '.join(body_text.split())
            article_data['body'] = body_text
        
        # Extract footnotes
        footnotes_elem = article_elem.find('footnotes')
        if footnotes_elem is not None:
            footnotes = []
            for footnote in footnotes_elem.findall('footnote'):
                footnote_id = footnote.get('id')
                footnote_text = footnote.text
                if footnote_id and footnote_text:
                    footnotes.append({
                        'id': int(footnote_id),
                        'text': ' '.join(footnote_text.strip().split())
                    })
            if footnotes:
                article_data['footnotes'] = footnotes
        
        # Extract bibliography
        biblio_elem = article_elem.find('bibliography')
        if biblio_elem is not None and biblio_elem.text:
            article_data['bibliography'] = biblio_elem.text.strip()
        
        return article_data

    def _xml_to_json(self, xml_string: str) -> str:
        """Convert XML output to JSON format."""
        # Wrap in root element if needed
        if not xml_string.strip().startswith('<?xml'):
            xml_string = f"<root>{xml_string}</root>"
        
        try:
            root = ET.fromstring(xml_string)
            
            # If we wrapped it, get the actual root
            if root.tag == 'root':
                elements = list(root)
            else:
                elements = [root]
            
            result = {
                'header': None,
                'articles': []
            }
            
            # Process all elements
            for elem in elements:
                if elem.tag == 'header':
                    result['header'] = elem.text.strip() if elem.text else ""
                elif elem.tag == 'article':
                    article_data = self._parse_article(elem)
                    # Skip empty continuation articles
                    if article_data.get('continuation') and not article_data.get('body'):
                        continue
                    result['articles'].append(article_data)
            
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        except ET.ParseError:
            # If parsing fails, return original
            return xml_string

    def __len__(self):
        base_len = len(self.examples)
        if self.augment_formats:
            base_len *= 2  # Each example has XML and JSON versions
        if self.augment_prompts and not self.randomize_prompts:
            base_len *= 2  # Each format has 2 prompt variations
        return base_len

    def __getitem__(self, idx):
        """Get an example by index."""
        # Calculate dimensions based on augmentation settings
        num_examples = len(self.examples)
        num_formats = 2 if self.augment_formats else 1
        num_prompts = 2 if self.augment_prompts and not self.randomize_prompts else 1
        
        # Decompose index into components
        if self.augment_prompts and not self.randomize_prompts and self.augment_formats:
            # Order: example -> format -> prompt
            prompt_idx = idx % num_prompts
            format_idx = (idx // num_prompts) % num_formats
            example_idx = idx // (num_prompts * num_formats)
        elif self.augment_formats:
            # Just format augmentation
            prompt_idx = 0
            format_idx = idx // num_examples
            example_idx = idx % num_examples
        else:
            # No augmentation
            prompt_idx = 0
            format_idx = 0
            example_idx = idx
            
        example = self.examples[example_idx]
        is_json = format_idx == 1
        
        # Load the actual image
        image = Image.open(example["image_path"]).convert("RGB")
        
        # Get appropriate prompt and output format
        if is_json:
            # Convert XML to JSON and use JSON prompt
            json_output = self._xml_to_json(example["xml_output"])
            if self.randomize_prompts:
                prompt = random.choice(self.JSON_PROMPTS)
            else:
                prompt = self.JSON_PROMPTS[prompt_idx % len(self.JSON_PROMPTS)]
            output_text = json_output
        else:
            # Use XML prompt and output
            if self.randomize_prompts:
                prompt = random.choice(self.XML_PROMPTS)
            else:
                prompt = self.XML_PROMPTS[prompt_idx % len(self.XML_PROMPTS)]
            output_text = example["xml_output"]
        
        # Format for training - image must come FIRST in content array
        formatted_example = {
            "id": f"{example['id']}_{'json' if is_json else 'xml'}_p{prompt_idx}",
            "images": [image],
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"{prompt}\n\nOCR Text:\n{example['ocr_text']}" if example['ocr_text'] else prompt}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": output_text}
                    ]
                }
            ]
        }
        
        return formatted_example

    def get_train_val_split(self, val_ratio: float = 0.1):
        """Split dataset into train and validation sets."""
        n_val = max(1, int(len(self.examples) * val_ratio))

        train_examples = self.examples[:-n_val]
        val_examples = self.examples[-n_val:]

        # Create new dataset instances
        train_dataset = OCRDataset.__new__(OCRDataset)
        train_dataset.ready_dir = self.ready_dir
        train_dataset.examples = train_examples
        train_dataset.randomize_prompts = self.randomize_prompts
        train_dataset.augment_formats = self.augment_formats
        train_dataset.augment_prompts = self.augment_prompts

        val_dataset = OCRDataset.__new__(OCRDataset)
        val_dataset.ready_dir = self.ready_dir
        val_dataset.examples = val_examples
        val_dataset.randomize_prompts = False  # Use consistent prompt for validation
        val_dataset.augment_formats = self.augment_formats
        val_dataset.augment_prompts = self.augment_prompts

        return train_dataset, val_dataset