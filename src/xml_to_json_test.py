#!/usr/bin/env python3
"""
Test script to convert XML output to JSON format on the fly.
This helps verify the model's output and test JSON schema variations.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
from typing import Dict, List, Any, Optional


def parse_article(article_elem: ET.Element) -> Dict[str, Any]:
    """Parse an article element into a dictionary."""
    article_data = {}
    
    # Handle continuation attribute
    if article_elem.get('continuation') == 'true':
        article_data['continuation'] = True
    
    # Extract title (optional)
    title_elem = article_elem.find('title')
    if title_elem is not None and title_elem.text:
        article_data['title'] = title_elem.text.strip()
    
    # Extract body (required)
    body_elem = article_elem.find('body')
    if body_elem is not None:
        # Get all text content recursively, preserving ref markers
        def get_all_text(elem):
            """Recursively get all text from an element, including ref markers."""
            parts = []
            if elem.text:
                parts.append(elem.text)
            for child in elem:
                if child.tag == 'ref':
                    # Include ref marker in text
                    ref_id = child.get('id', '?')
                    parts.append(f'[{ref_id}]')
                else:
                    parts.extend(get_all_text(child))
                if child.tail:
                    parts.append(child.tail)
            return parts
        
        # Also extract refs separately for footnote_refs list
        refs = []
        for ref in body_elem.findall('.//ref'):
            ref_id = ref.get('id')
            if ref_id:
                refs.append(int(ref_id))
        
        # Get all text and normalize whitespace
        text_parts = get_all_text(body_elem)
        body_text = ' '.join(text_parts)
        # Normalize whitespace - replace multiple spaces/newlines with single space
        body_text = ' '.join(body_text.split())
        
        article_data['body'] = body_text
        if refs:
            article_data['footnote_refs'] = refs
    
    # Extract footnotes (optional)
    footnotes_elem = article_elem.find('footnotes')
    if footnotes_elem is not None:
        footnotes = []
        for footnote in footnotes_elem.findall('footnote'):
            footnote_id = footnote.get('id')
            footnote_text = footnote.text
            if footnote_id and footnote_text:
                footnotes.append({
                    'id': int(footnote_id),
                    'text': footnote_text.strip()
                })
        if footnotes:
            article_data['footnotes'] = footnotes
    
    # Extract bibliography (optional)
    biblio_elem = article_elem.find('bibliography')
    if biblio_elem is not None and biblio_elem.text:
        article_data['bibliography'] = biblio_elem.text.strip()
    
    return article_data


def xml_to_json(xml_string: str) -> Dict[str, Any]:
    """Convert encyclopedia XML to JSON format."""
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
                article_data = parse_article(elem)
                result['articles'].append(article_data)
        
        return result
        
    except ET.ParseError as e:
        return {'error': f'XML parse error: {str(e)}'}


def test_conversion(xml_file: Path) -> None:
    """Test XML to JSON conversion on a file."""
    print(f"\nTesting: {xml_file}")
    print("-" * 60)
    
    with open(xml_file, 'r', encoding='utf-8') as f:
        xml_content = f.read()
    
    json_result = xml_to_json(xml_content)
    
    print(json.dumps(json_result, indent=2, ensure_ascii=False))
    
    # Basic validation
    if 'error' in json_result:
        print(f"\n❌ Error: {json_result['error']}")
    else:
        print(f"\n✓ Header: {'Present' if json_result['header'] else 'Missing'}")
        print(f"✓ Articles: {len(json_result['articles'])}")
        for i, article in enumerate(json_result['articles']):
            print(f"  Article {i+1}:")
            print(f"    - Title: {'Present' if 'title' in article else 'Missing'}")
            print(f"    - Continuation: {article.get('continuation', False)}")
            print(f"    - Body length: {len(article.get('body', ''))}")
            print(f"    - Footnotes: {len(article.get('footnotes', []))}")
            print(f"    - Bibliography: {'Present' if 'bibliography' in article else 'Missing'}")


def main():
    parser = argparse.ArgumentParser(description="Convert encyclopedia XML to JSON")
    parser.add_argument("xml_file", type=Path, nargs='?', 
                       help="XML file to convert (optional - will test all in ready if not provided)")
    parser.add_argument("--ready-dir", type=Path, default=Path("ready"),
                       help="Directory with XML files")
    parser.add_argument("--output", "-o", type=Path,
                       help="Save JSON output to file")
    
    args = parser.parse_args()
    
    if args.xml_file:
        # Test single file
        test_conversion(args.xml_file)
    else:
        # Test all XML files in ready directory
        xml_files = []
        for example_dir in sorted(args.ready_dir.iterdir()):
            if example_dir.is_dir():
                for xml_file in example_dir.glob("*_done.xml"):
                    xml_files.append(xml_file)
        
        print(f"Found {len(xml_files)} XML files to test")
        
        for xml_file in xml_files[:3]:  # Test first 3
            test_conversion(xml_file)


if __name__ == "__main__":
    main()