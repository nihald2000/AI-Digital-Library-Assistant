# chunker.py 
import logging
from typing import List, Dict, Any, Optional
import re
from .models import Chunk
from .text_preprocessor import TextPreprocessor
import config

logger = logging.getLogger(__name__)

class TextChunker:
    def __init__(self):
        self.config = config.config
        self.preprocessor = TextPreprocessor()
        
        self.chunk_size = self.config.CHUNK_SIZE
        self.chunk_overlap = self.config.CHUNK_OVERLAP
    
    def chunk_document(self, document_id: str, content: str, method: str = "recursive") -> List[Chunk]:
        """Chunk a document using the specified method"""
        if not content:
            return []
        
        try:
            if method == "recursive":
                return self._recursive_chunk(document_id, content)
            elif method == "sentence":
                return self._sentence_chunk(document_id, content)
            elif method == "paragraph":
                return self._paragraph_chunk(document_id, content)
            elif method == "fixed":
                return self._fixed_chunk(document_id, content)
            else:
                logger.warning(f"Unknown chunking method: {method}, using recursive")
                return self._recursive_chunk(document_id, content)
        except Exception as e:
            logger.error(f"Error chunking document: {str(e)}")
            # Fallback to simple fixed chunking
            return self._fixed_chunk(document_id, content)
    
    def _recursive_chunk(self, document_id: str, content: str) -> List[Chunk]:
        """Recursively split text by different separators"""
        chunks = []
        
        # Define separators in order of preference
        separators = [
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            ", ",    # Clauses
            " "      # Words
        ]
        
        def split_text(text: str, separators: List[str], chunk_size: int) -> List[str]:
            if len(text) <= chunk_size:
                return [text] if text.strip() else []
            
            if not separators:
                # If no separators left, split by character
                return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            
            separator = separators[0]
            remaining_separators = separators[1:]
            
            splits = text.split(separator)
            result = []
            current_chunk = ""
            
            for split in splits:
                if len(current_chunk) + len(split) + len(separator) <= chunk_size:
                    if current_chunk:
                        current_chunk += separator + split
                    else:
                        current_chunk = split
                else:
                    if current_chunk:
                        result.append(current_chunk)
                    
                    if len(split) > chunk_size:
                        # Split is too big, need to split further
                        result.extend(split_text(split, remaining_separators, chunk_size))
                        current_chunk = ""
                    else:
                        current_chunk = split
            
            if current_chunk:
                result.append(current_chunk)
            
            return result
        
        text_chunks = split_text(content, separators, self.chunk_size)

        # Position mapping via a monotonic cursor: each chunk is searched
        # forward from the previous chunk's start. This turns an O(N*M) full
        # scan into ~O(len(content)) total work at 1M-chunk scale.
        chunks = []
        cursor = 0
        for i, raw_chunk in enumerate(text_chunks):
            raw_text = raw_chunk.strip()
            if not raw_text:
                continue

            if i > 0 and self.chunk_overlap > 0:
                prev_chunk = text_chunks[i - 1]
                overlap_text = (
                    prev_chunk[-self.chunk_overlap:]
                    if len(prev_chunk) > self.chunk_overlap
                    else prev_chunk
                )
                chunk_text = overlap_text + " " + raw_chunk
            else:
                chunk_text = raw_chunk

            start_pos = content.find(raw_text, cursor)
            if start_pos == -1:
                # overlap-prepended or whitespace-collapsed text can drift;
                # degrade to cursor-based estimate instead of a rescan whole doc
                start_pos = cursor
            else:
                cursor = start_pos + len(raw_text)
            end_pos = start_pos + len(raw_text)

            chunks.append(Chunk(
                id=self._generate_chunk_id(document_id, len(chunks)),
                document_id=document_id,
                content=chunk_text.strip(),
                chunk_index=len(chunks),
                start_pos=start_pos,
                end_pos=end_pos,
                metadata={
                    "chunk_method": "recursive",
                    "original_length": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                },
            ))

        return chunks
    
    def _sentence_chunk(self, document_id: str, content: str) -> List[Chunk]:
        """Chunk text by sentences"""
        chunks = []
        sentences = self.preprocessor.extract_sentences(content)
        
        current_chunk = ""
        chunk_index = 0
        start_pos = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                    start_pos = content.find(sentence)
            else:
                if current_chunk:
                    chunk = Chunk(
                        id=self._generate_chunk_id(document_id, chunk_index),
                        document_id=document_id,
                        content=current_chunk.strip(),
                        chunk_index=chunk_index,
                        start_pos=start_pos,
                        end_pos=start_pos + len(current_chunk),
                        metadata={
                            "chunk_method": "sentence",
                            "sentence_count": len(self.preprocessor.extract_sentences(current_chunk))
                        }
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                current_chunk = sentence
                start_pos = content.find(sentence)
        
        # Add final chunk
        if current_chunk:
            chunk = Chunk(
                id=self._generate_chunk_id(document_id, chunk_index),
                document_id=document_id,
                content=current_chunk.strip(),
                chunk_index=chunk_index,
                start_pos=start_pos,
                end_pos=start_pos + len(current_chunk),
                metadata={
                    "chunk_method": "sentence",
                    "sentence_count": len(self.preprocessor.extract_sentences(current_chunk))
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _paragraph_chunk(self, document_id: str, content: str) -> List[Chunk]:
        """Chunk text by paragraphs"""
        chunks = []
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        current_chunk = ""
        chunk_index = 0
        start_pos = 0
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                    start_pos = content.find(paragraph)
            else:
                if current_chunk:
                    chunk = Chunk(
                        id=self._generate_chunk_id(document_id, chunk_index),
                        document_id=document_id,
                        content=current_chunk.strip(),
                        chunk_index=chunk_index,
                        start_pos=start_pos,
                        end_pos=start_pos + len(current_chunk),
                        metadata={
                            "chunk_method": "paragraph",
                            "paragraph_count": len([p for p in current_chunk.split('\n\n') if p.strip()])
                        }
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # If paragraph is too long, split it further
                if len(paragraph) > self.chunk_size:
                    para_chunks = self._fixed_chunk(document_id, paragraph)
                    for pc in para_chunks:
                        pc.chunk_index = chunk_index
                        pc.id = self._generate_chunk_id(document_id, chunk_index)
                        chunks.append(pc)
                        chunk_index += 1
                else:
                    current_chunk = paragraph
                    start_pos = content.find(paragraph)
        
        # Add final chunk
        if current_chunk:
            chunk = Chunk(
                id=self._generate_chunk_id(document_id, chunk_index),
                document_id=document_id,
                content=current_chunk.strip(),
                chunk_index=chunk_index,
                start_pos=start_pos,
                end_pos=start_pos + len(current_chunk),
                metadata={
                    "chunk_method": "paragraph",
                    "paragraph_count": len([p for p in current_chunk.split('\n\n') if p.strip()])
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _fixed_chunk(self, document_id: str, content: str) -> List[Chunk]:
        """Simple fixed-size chunking with overlap"""
        chunks = []
        
        for i in range(0, len(content), self.chunk_size - self.chunk_overlap):
            chunk_text = content[i:i + self.chunk_size]
            
            if not chunk_text.strip():
                continue
            
            chunk = Chunk(
                id=self._generate_chunk_id(document_id, len(chunks)),
                document_id=document_id,
                content=chunk_text.strip(),
                chunk_index=len(chunks),
                start_pos=i,
                end_pos=min(i + self.chunk_size, len(content)),
                metadata={
                    "chunk_method": "fixed",
                    "original_length": len(chunk_text)
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _generate_chunk_id(self, document_id: str, chunk_index: int) -> str:
        """Generate a unique chunk ID"""
        return f"{document_id}_chunk_{chunk_index}"
    
    def optimize_chunks_for_embedding(self, chunks: List[Chunk]) -> List[Chunk]:
        """Optimize chunks for better embedding generation"""
        optimized_chunks = []
        
        for chunk in chunks:
            # Clean the content for embedding
            clean_content = self.preprocessor.prepare_for_embedding(chunk.content)
            
            # Skip very short chunks
            if len(clean_content.split()) < 5:
                continue
            
            # Update chunk with optimized content
            optimized_chunk = Chunk(
                id=chunk.id,
                document_id=chunk.document_id,
                content=clean_content,
                chunk_index=chunk.chunk_index,
                start_pos=chunk.start_pos,
                end_pos=chunk.end_pos,
                metadata={
                    **chunk.metadata,
                    "optimized_for_embedding": True,
                    "original_content_length": len(chunk.content),
                    "optimized_content_length": len(clean_content)
                }
            )
            optimized_chunks.append(optimized_chunk)
        
        return optimized_chunks