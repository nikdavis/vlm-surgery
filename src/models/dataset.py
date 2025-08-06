"""
Pydantic models for the unified dataset schema.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, ConfigDict


class SchemaType(str, Enum):
    """Supported schema types for structured extraction."""
    JSON = "json"
    XML = "xml"


class OutputFormat(str, Enum):
    """Output format types."""
    TEXT = "text"
    JSON = "json"
    XML = "xml"


class DataSource(str, Enum):
    """Data source types for provenance tracking."""
    MANUAL = "manual"          # Hand-crafted examples
    SYNTHETIC = "synthetic"    # AI-generated
    OCR = "ocr"               # From OCR extraction
    CONVERTED = "converted"    # Converted from legacy format
    MIXED = "mixed"           # Mixed sources


class Schema(BaseModel):
    """Schema definition for structured extraction tasks."""
    type: SchemaType
    definition: str = Field(..., description="JSON Schema or XSD definition")
    version: Optional[str] = Field(None, description="Schema version")


class Response(BaseModel):
    """Model response with optional chain of thought."""
    thinking: Optional[str] = Field(
        None, 
        description="Chain of thought reasoning (will be wrapped in <thk> tags)"
    )
    output: str = Field(..., description="Final output/answer")
    format: OutputFormat = Field(
        OutputFormat.TEXT,
        description="Output format hint"
    )


class Provenance(BaseModel):
    """Track data origin and creation details."""
    source: DataSource
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = Field(None, description="Model or person who created this")
    synthetic_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Configuration used for synthetic generation"
    )
    conversion_notes: Optional[str] = Field(
        None,
        description="Notes about data conversion or processing"
    )
    original_format: Optional[str] = Field(
        None,
        description="Original data format before conversion"
    )


class Metadata(BaseModel):
    """Additional metadata for evaluation and tracking."""
    ground_truth: Optional[str] = Field(None, description="Expected answer for evaluation")
    model_used: Optional[str] = Field(None, description="Model used to generate response")
    response_time: Optional[float] = Field(None, description="Response generation time in seconds")
    processed_at: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    difficulty: Optional[Literal["easy", "medium", "hard"]] = None
    notes: Optional[str] = Field(None, description="Additional notes")


class DatasetExample(BaseModel):
    """Single example in the unified dataset format."""
    model_config = ConfigDict(extra="forbid")
    
    # Core fields
    id: str = Field(..., description="Unique identifier")
    images: List[str] = Field(..., description="List of image file paths")
    prompt: str = Field(..., description="User prompt/question")
    
    # Optional fields
    schema_: Optional[Schema] = Field(None, alias="schema", description="Schema for structured extraction")
    ocr_text: Optional[str] = Field(None, description="Pre-extracted OCR text")
    
    # Response
    response: Response
    
    # Tracking
    metadata: Optional[Metadata] = Field(default_factory=Metadata)
    provenance: Provenance
    
    # Dataset version
    version: str = Field("1.0", description="Dataset schema version")


class Dataset(BaseModel):
    """Complete dataset with multiple examples."""
    model_config = ConfigDict(extra="forbid")
    
    version: str = Field("1.0", description="Dataset format version")
    name: str = Field(..., description="Dataset name")
    description: Optional[str] = Field(None, description="Dataset description")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    examples: List[DatasetExample]
    
    # Dataset-level metadata
    stats: Optional[Dict[str, Any]] = Field(None, description="Dataset statistics")
    license: Optional[str] = Field(None, description="Dataset license")
    citation: Optional[str] = Field(None, description="How to cite this dataset")


# Convenience functions for common operations
def create_manual_example(
    id: str,
    images: List[str],
    prompt: str,
    output: str,
    thinking: Optional[str] = None,
    ocr_text: Optional[str] = None,
    ground_truth: Optional[str] = None,
    notes: Optional[str] = None
) -> DatasetExample:
    """Create a manually crafted example."""
    return DatasetExample(
        id=id,
        images=images,
        prompt=prompt,
        ocr_text=ocr_text,
        response=Response(
            thinking=thinking,
            output=output,
            format=OutputFormat.TEXT
        ),
        metadata=Metadata(
            ground_truth=ground_truth,
            notes=notes
        ),
        provenance=Provenance(
            source=DataSource.MANUAL,
            created_by="human"
        )
    )


def create_synthetic_example(
    id: str,
    images: List[str],
    prompt: str,
    response_text: str,
    model_used: str,
    response_time: float,
    ground_truth: Optional[str] = None
) -> DatasetExample:
    """Create a synthetic example from model generation."""
    # Try to extract thinking from <thk> tags
    import re
    thinking_match = re.search(r'<thk>(.*?)</thk>', response_text, re.DOTALL)
    if thinking_match:
        thinking = thinking_match.group(1).strip()
        output = response_text[thinking_match.end():].strip()
    else:
        thinking = None
        output = response_text
    
    # Detect format
    if output.strip().startswith("<") and ">" in output:
        format = OutputFormat.XML
    elif output.strip().startswith("{") and output.strip().endswith("}"):
        format = OutputFormat.JSON
    else:
        format = OutputFormat.TEXT
    
    return DatasetExample(
        id=id,
        images=images,
        prompt=prompt,
        response=Response(
            thinking=thinking,
            output=output,
            format=format
        ),
        metadata=Metadata(
            ground_truth=ground_truth,
            model_used=model_used,
            response_time=response_time,
            processed_at=datetime.utcnow()
        ),
        provenance=Provenance(
            source=DataSource.SYNTHETIC,
            created_by=model_used
        )
    )