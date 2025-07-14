import re
import time
import logging
from typing import Dict, Any, Optional, Tuple, Union
from enum import Enum
import tempfile
import os
from pathlib import Path

import weasyprint
import mistune
from mistune import create_markdown
# Import only the core available plugins
try:
    # These are the built-in plugins for mistune
    AVAILABLE_PLUGINS = [
        "footnotes",
        "table",
        "abbr",
        "def_list",
        "task_lists",
        "math",
        "url",
    ]
except ImportError:
    AVAILABLE_PLUGINS = []
    
import pypandoc
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

from .metrics import ResourceTracker

logger = logging.getLogger(__name__)

class Priority(str, Enum):
    AUTO = "auto"
    SPEED = "speed"
    QUALITY = "quality"

class ContentType(str, Enum):
    HTML = "html"
    MARKDOWN = "markdown"

# Enhanced markdown patterns for better complexity detection
COMPLEX_MARKDOWN_PATTERNS = [
    r"\\\[.*?\\\]",              # LaTeX math blocks
    r"\\\(.*?\\\)",              # Inline LaTeX math
    r"\[\^.+?\]",                # Footnotes
    r"\[\[.+?\]\]",              # Citations
    r"^\s*```",                  # Fenced code blocks
    r"\{\s*\.?[^}]+\}",          # Raw attributes (pandoc-style)
    r":::",                      # Admonitions (common in docs)
    r"^\s*\|.*\|.*\|",           # Complex tables
    r"^\s*[-:]+\s*\|",           # Table separators
    r"^\s*>\s*>",                # Nested blockquotes
    r"^\s*\d+\.\s+",             # Numbered lists (can be complex)
    r"^\s*-\s+\[[ x]\]",         # Task lists
    r"!\[.*?\]\(.*?\)",          # Images with complex paths
    r"\[.*?\]\(.*?\s+\".*?\"\)", # Links with titles
    r"^\s*##{6,}",               # Deep heading levels
    r"<[^>]+>",                  # HTML tags mixed in
    r"^\s*\*\*\*+\s*$",          # Horizontal rules
    r"^\s*---+\s*$",             # YAML front matter separators
    r"^\s*\.\.\.",                # Ellipsis patterns
    r"\$\$.*?\$\$",              # Math blocks
    r"\$.*?\$",                  # Inline math
    r"~~.*?~~",                  # Strikethrough
    r"==.*?==",                  # Mark/highlight
    r"\^\^.*?\^\^",              # Superscript
    r"~.*?~",                    # Subscript
]

class PDFConverter:
    def __init__(self):
        self.font_config = FontConfiguration()
        
        # Initialize enhanced mistune markdown processor with available plugins
        try:
            self.mistune_markdown = create_markdown(
                escape=False,
                plugins=AVAILABLE_PLUGINS
            )
        except Exception as e:
            logger.warning(f"Failed to initialize enhanced mistune: {e}, using basic mistune")
            self.mistune_markdown = None

    def detect_complexity(self, content: str, content_type: ContentType) -> Tuple[bool, list]:
        """
        Detect if content has complex features that require pandoc
        
        Args:
            content: The content to analyze
            content_type: Type of content
            
        Returns:
            Tuple of (is_complex, matched_patterns)
        """
        if content_type == ContentType.HTML:
            # For HTML, check for complex tags or structures
            html_complex_patterns = [
                r"<math.*?>.*?</math>",      # MathML
                r"<table.*?>.*?</table>",    # Complex tables
                r"<svg.*?>.*?</svg>",        # SVG graphics
                r"<canvas.*?>.*?</canvas>",  # Canvas elements
                r"<script.*?>.*?</script>",  # Scripts
                r"<style.*?>.*?</style>",    # Embedded styles
                r"<iframe.*?>.*?</iframe>",  # Iframes
                r"<object.*?>.*?</object>",  # Objects
                r"<embed.*?>.*?</embed>",    # Embedded content
            ]
            matched_patterns = []
            for pattern in html_complex_patterns:
                if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                    matched_patterns.append(pattern)
            return len(matched_patterns) > 0, matched_patterns
        
        # For Markdown, check for complex patterns
        matched_patterns = []
        for pattern in COMPLEX_MARKDOWN_PATTERNS:
            if re.search(pattern, content, re.MULTILINE):
                matched_patterns.append(pattern)
        
        # Consider complex if more than 2 patterns match
        return len(matched_patterns) > 2, matched_patterns

    def choose_converter(self, content: str, content_type: ContentType, priority: Priority) -> str:
        """
        Choose between mistune and pandoc based on complexity and priority
        
        Args:
            content: The content to convert
            content_type: Type of content
            priority: Processing priority
            
        Returns:
            Chosen converter ("mistune" or "pandoc")
        """
        is_complex, patterns = self.detect_complexity(content, content_type)
        
        if priority == Priority.SPEED:
            # Always prefer mistune for speed
            return "mistune"
        elif priority == Priority.QUALITY:
            # Always prefer pandoc for quality
            return "pandoc"
        elif priority == Priority.AUTO:
            # Auto-detect based on complexity
            if is_complex:
                logger.info(f"Content complexity detected ({len(patterns)} patterns), using pandoc")
                return "pandoc"
            else:
                logger.info("Content is simple, using mistune for speed")
                return "mistune"
        
        return "mistune"  # Default fallback
    
    def convert_markdown_with_mistune(self, content: str) -> str:
        """
        Convert markdown to HTML using enhanced mistune with plugins
        
        Args:
            content: Markdown content
            
        Returns:
            HTML string
        """
        try:
            # Use enhanced mistune with plugins if available
            if self.mistune_markdown:
                result = self.mistune_markdown(content)
                return str(result) if result else ""
            else:
                # Fallback to basic mistune
                return str(mistune.html(content))
        except Exception as e:
            logger.warning(f"Enhanced mistune conversion failed: {e}, falling back to simple mistune")
            # Fallback to simple mistune
            return str(mistune.html(content))

    def convert_markdown_with_pandoc(self, content: str) -> str:
        """
        Convert markdown to HTML using pandoc
        
        Args:
            content: Markdown content
            
        Returns:
            HTML string
        """
        try:
            # Enhanced pandoc options for better output
            extra_args = [
                '--from=markdown+smart+footnotes+fancy_lists+definition_lists+table_captions+yaml_metadata_block+tex_math_dollars',
                '--to=html5',
                '--mathjax',  # Use MathJax for math rendering
                '--highlight-style=pygments',  # Syntax highlighting
                '--wrap=none',  # Don't wrap lines
                '--standalone'  # Generate complete HTML
            ]
            
            html_content = pypandoc.convert_text(
                content,
                'html',
                format='markdown',
                extra_args=extra_args
            )
            return str(html_content) if html_content else ""
        except Exception as e:
            logger.warning(f"Pandoc conversion failed: {e}, falling back to mistune")
            return self.convert_markdown_with_mistune(content)

    def create_pdf_from_html(self, html_content: str, css_content: Optional[str] = None, 
                           weasyprint_options: Optional[Dict[str, Any]] = None) -> bytes:
        """
        Create PDF from HTML using weasyprint
        
        Args:
            html_content: HTML content to convert
            css_content: Optional CSS content
            weasyprint_options: Optional weasyprint configuration
            
        Returns:
            PDF bytes
        """
        options = weasyprint_options or {}
        
        # Default CSS for better PDF formatting
        default_css = """
        @page {
            size: A4;
            margin: 1in;
        }
        body {
            font-family: 'DejaVu Sans', Arial, sans-serif;
            font-size: 12pt;
            line-height: 1.4;
            color: #333;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50;
            page-break-after: avoid;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 1em;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'DejaVu Sans Mono', monospace;
        }
        pre {
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }
        blockquote {
            border-left: 4px solid #ccc;
            margin-left: 0;
            padding-left: 1em;
            color: #666;
        }
        """
        
        if css_content:
            css_content = default_css + "\n" + css_content
        else:
            css_content = default_css
        
        # Create HTML document
        html_doc = HTML(string=html_content, base_url=".")
        css_doc = CSS(string=css_content, font_config=self.font_config)
        
        # Generate PDF
        pdf_bytes = html_doc.write_pdf(stylesheets=[css_doc], **options)
        if pdf_bytes is None:
            raise Exception("PDF generation failed - weasyprint returned None")
        return pdf_bytes

    async def convert_to_pdf(self, 
                           content: str, 
                           content_type: ContentType,
                           priority: Priority = Priority.AUTO,
                           css_content: Optional[str] = None,
                           weasyprint_options: Optional[Dict[str, Any]] = None,
                           tracker: Optional[ResourceTracker] = None) -> Tuple[bytes, Dict[str, Any]]:
        """
        Convert HTML or Markdown content to PDF
        
        Args:
            content: The content to convert
            content_type: Type of content (html/markdown)
            priority: Processing priority
            css_content: Optional CSS content
            weasyprint_options: Optional weasyprint configuration
            tracker: Optional resource tracker
            
        Returns:
            Tuple of (PDF bytes, conversion stats)
        """
        start_time = time.time()
        stats = {
            "content_type": content_type.value,
            "priority": priority.value,
            "content_length": len(content),
        }
        
        if tracker:
            tracker.log_method(f"pdf_conversion_{content_type.value}")
        
        try:
            # If content is HTML, use it directly
            if content_type == ContentType.HTML:
                html_content = content
                stats["converter_used"] = "direct_html"
            else:
                # For Markdown, choose converter based on complexity and priority
                converter_name = self.choose_converter(content, content_type, priority)
                stats["converter_used"] = converter_name
                
                if converter_name == "pandoc":
                    html_content = self.convert_markdown_with_pandoc(content)
                else:
                    html_content = self.convert_markdown_with_mistune(content)
            
            # Detect complexity for stats
            is_complex, patterns = self.detect_complexity(content, content_type)
            stats["complexity_detected"] = is_complex
            stats["complexity_patterns_count"] = len(patterns)
            
            # Convert to PDF
            pdf_bytes = self.create_pdf_from_html(html_content, css_content, weasyprint_options)
            
            # Calculate stats
            end_time = time.time()
            stats.update({
                "conversion_time_seconds": round(end_time - start_time, 3),
                "html_length": len(html_content),
                "pdf_size_bytes": len(pdf_bytes),
                "success": True
            })
            
            logger.info(f"PDF conversion completed: {stats}")
            return pdf_bytes, stats
            
        except Exception as e:
            end_time = time.time()
            stats.update({
                "conversion_time_seconds": round(end_time - start_time, 3),
                "success": False,
                "error": str(e)
            })
            logger.error(f"PDF conversion failed: {e}")
            raise 