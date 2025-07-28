"""
Pydantic-XML models for different schema variations of OCR output.
Supports fuzzing between different tag names and structures.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel
from pydantic_xml import BaseXmlModel, RootXmlModel, attr, element


# Schema 1: Original schema (article/body/ref)
class Schema1Footnote(BaseXmlModel, tag='footnote'):
    id: int = attr()
    text: str


class Schema1Article(BaseXmlModel, tag='article'):
    continuation: Optional[Literal['true']] = attr(default=None)
    title: Optional[str] = element(default=None)
    body: str = element()
    footnotes: Optional[List[Schema1Footnote]] = element(tag='footnote', default=None)
    bibliography: Optional[str] = element(default=None)


class Schema1Document(BaseXmlModel, tag='document'):
    header: str = element()
    articles: List[Schema1Article] = element(tag='article')


# Schema 2: Alternative naming (entry/content/citation)
class Schema2Note(BaseXmlModel, tag='note'):
    id: int = attr()
    text: str


class Schema2Entry(BaseXmlModel, tag='entry'):
    continued: Optional[Literal['true']] = attr(default=None)
    heading: Optional[str] = element(default=None)
    content: str = element()
    notes: Optional[List[Schema2Note]] = element(tag='note', default=None)
    references: Optional[str] = element(default=None)


class Schema2Document(BaseXmlModel, tag='page'):
    header: str = element()
    entries: List[Schema2Entry] = element(tag='entry')


# Schema 3: Semantic naming (section/text/annotation)
class Schema3Annotation(BaseXmlModel, tag='annotation'):
    id: int = attr()
    text: str


class Schema3Section(BaseXmlModel, tag='section'):
    incomplete: Optional[Literal['true']] = attr(default=None)
    caption: Optional[str] = element(default=None)
    text: str = element()
    annotations: Optional[List[Schema3Annotation]] = element(tag='annotation', default=None)
    sources: Optional[str] = element(default=None)


class Schema3Document(BaseXmlModel, tag='content'):
    header: str = element()
    sections: List[Schema3Section] = element(tag='section')


class SchemaMapper:
    """Maps between different schema representations."""
    
    @staticmethod
    def schema1_to_schema2(doc1: Schema1Document) -> Schema2Document:
        """Convert Schema1 to Schema2."""
        entries = []
        for article in doc1.articles:
            entry = Schema2Entry(
                continued='true' if article.continuation else None,
                heading=article.title,
                content=article.body,
                references=article.bibliography
            )
            if article.footnotes:
                entry.notes = [
                    Schema2Note(id=fn.id, text=fn.text)
                    for fn in article.footnotes
                ]
            entries.append(entry)
        
        return Schema2Document(header=doc1.header, entries=entries)
    
    @staticmethod
    def schema1_to_schema3(doc1: Schema1Document) -> Schema3Document:
        """Convert Schema1 to Schema3."""
        sections = []
        for article in doc1.articles:
            section = Schema3Section(
                incomplete='true' if article.continuation else None,
                caption=article.title,
                text=article.body,
                sources=article.bibliography
            )
            if article.footnotes:
                section.annotations = [
                    Schema3Annotation(id=fn.id, text=fn.text)
                    for fn in article.footnotes
                ]
            sections.append(section)
        
        return Schema3Document(header=doc1.header, sections=sections)
    
    @staticmethod
    def process_body_with_refs(body: str, footnotes: Optional[List]) -> str:
        """Keep [N] markers for both XML and JSON."""
        # No conversion needed - keep [N] format
        return body
    
    @staticmethod
    def process_content_with_citations(content: str, notes: Optional[List]) -> str:
        """Keep [N] markers for both XML and JSON."""
        # No conversion needed - keep [N] format
        return content
    
    @staticmethod
    def process_text_with_marks(text: str, annotations: Optional[List]) -> str:
        """Keep [N] markers for both XML and JSON."""
        # No conversion needed - keep [N] format
        return text