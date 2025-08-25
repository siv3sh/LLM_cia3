"""
Vector Store for storing and retrieving embeddings
"""

import asyncio
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import json
import pickle
from datetime import datetime

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from core.config import Config


class VectorStore:
    """
    Manages vector storage and retrieval for RAG functionality
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Vector store components
        self.client: Optional[chromadb.Client] = None
        self.collection: Optional[chromadb.Collection] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        
        # Store metadata
        self.collection_name = "attrition_analysis"
        self.metadata: Dict[str, Any] = {}
        
        # Performance tracking
        self.query_stats: Dict[str, Any] = {}
        self.index_stats: Dict[str, Any] = {}
    
    async def initialize(self):
        """Initialize the vector store"""
        try:
            # Create vector store directory
            vector_store_path = Path(self.config.vector_store_path)
            vector_store_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(vector_store_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(self.config.vector_embedding_model)
            
            # Create or get collection
            try:
                self.collection = self.client.get_collection(self.collection_name)
                self.logger.info(f"Retrieved existing collection: {self.collection_name}")
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Attrition analysis documents and insights"}
                )
                self.logger.info(f"Created new collection: {self.collection_name}")
            
            # Load metadata if exists
            metadata_file = vector_store_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            
            self.logger.info("Vector store initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add documents to the vector store"""
        try:
            if not documents:
                raise ValueError("No documents provided")
            
            self.logger.info(f"Adding {len(documents)} documents to vector store")
            
            # Prepare documents for indexing
            texts = []
            metadatas = []
            ids = []
            
            for i, doc in enumerate(documents):
                # Extract text content
                if isinstance(doc, dict):
                    text = doc.get('content', doc.get('text', str(doc)))
                    metadata = doc.get('metadata', {})
                    doc_id = doc.get('id', f"doc_{i}_{datetime.now().timestamp()}")
                else:
                    text = str(doc)
                    metadata = {}
                    doc_id = f"doc_{i}_{datetime.now().timestamp()}"
                
                # Clean and validate text
                if not text or len(text.strip()) < 10:
                    self.logger.warning(f"Skipping document {doc_id}: text too short")
                    continue
                
                texts.append(text)
                metadatas.append(metadata)
                ids.append(doc_id)
            
            if not texts:
                raise ValueError("No valid texts found in documents")
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts)
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            # Update metadata
            self.metadata["total_documents"] = self.metadata.get("total_documents", 0) + len(texts)
            self.metadata["last_updated"] = datetime.utcnow().isoformat()
            self.metadata["document_types"] = list(set(
                meta.get("type", "unknown") for meta in metadatas
            ))
            
            # Save metadata
            await self._save_metadata()
            
            # Update index stats
            self.index_stats["last_indexing"] = {
                "timestamp": datetime.utcnow().isoformat(),
                "documents_added": len(texts),
                "total_documents": self.metadata["total_documents"]
            }
            
            self.logger.info(f"Successfully added {len(texts)} documents to vector store")
            
            return {
                "success": True,
                "documents_added": len(texts),
                "total_documents": self.metadata["total_documents"],
                "collection_name": self.collection_name
            }
            
        except Exception as e:
            self.logger.error(f"Failed to add documents: {e}")
            raise
    
    async def search_documents(self, query: str, top_k: int = 5, 
                              filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")
            
            self.logger.info(f"Searching for: '{query}' (top_k={top_k})")
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Perform search
            search_results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k,
                where=filters
            )
            
            # Format results
            results = []
            if search_results['documents'] and search_results['documents'][0]:
                for i in range(len(search_results['documents'][0])):
                    result = {
                        'id': search_results['ids'][0][i],
                        'content': search_results['documents'][0][i],
                        'metadata': search_results['metadatas'][0][i] if search_results['metadatas'] else {},
                        'distance': search_results['distances'][0][i] if search_results['distances'] else None,
                        'similarity_score': 1 - (search_results['distances'][0][i] if search_results['distances'] else 0)
                    }
                    results.append(result)
            
            # Update query stats
            self.query_stats["last_query"] = {
                "timestamp": datetime.utcnow().isoformat(),
                "query": query,
                "top_k": top_k,
                "results_returned": len(results),
                "filters": filters
            }
            
            self.logger.info(f"Search completed: {len(results)} results found")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Document search failed: {e}")
            raise
    
    async def semantic_search(self, query: str, top_k: int = 5, 
                            threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Perform semantic search with similarity threshold"""
        try:
            # Get initial search results
            results = await self.search_documents(query, top_k=top_k * 2)
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in results
                if result['similarity_score'] >= threshold
            ]
            
            # Sort by similarity score
            filtered_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Return top_k results
            return filtered_results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            raise
    
    async def get_similar_documents(self, document_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find documents similar to a specific document"""
        try:
            # Get the document
            document = self.collection.get(ids=[document_id])
            
            if not document['documents']:
                raise ValueError(f"Document {document_id} not found")
            
            # Use the document content as query
            query = document['documents'][0]
            
            # Search for similar documents
            results = await self.search_documents(query, top_k=top_k + 1)
            
            # Remove the original document from results
            results = [r for r in results if r['id'] != document_id]
            
            return results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Similar document search failed: {e}")
            raise
    
    async def update_document(self, document_id: str, new_content: str, 
                            new_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Update an existing document"""
        try:
            # Generate new embedding
            new_embedding = self.embedding_model.encode([new_content])
            
            # Update the document
            self.collection.update(
                ids=[document_id],
                embeddings=new_embedding.tolist(),
                documents=[new_content],
                metadatas=[new_metadata] if new_metadata else None
            )
            
            # Update metadata
            self.metadata["last_updated"] = datetime.utcnow().isoformat()
            await self._save_metadata()
            
            self.logger.info(f"Document {document_id} updated successfully")
            
            return {
                "success": True,
                "document_id": document_id,
                "message": "Document updated successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to update document: {e}")
            raise
    
    async def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete a document from the vector store"""
        try:
            # Delete the document
            self.collection.delete(ids=[document_id])
            
            # Update metadata
            self.metadata["total_documents"] = max(0, self.metadata.get("total_documents", 1) - 1)
            self.metadata["last_updated"] = datetime.utcnow().isoformat()
            await self._save_metadata()
            
            self.logger.info(f"Document {document_id} deleted successfully")
            
            return {
                "success": True,
                "document_id": document_id,
                "message": "Document deleted successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to delete document: {e}")
            raise
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            collection_info = {
                "name": self.collection_name,
                "total_documents": self.metadata.get("total_documents", 0),
                "last_updated": self.metadata.get("last_updated"),
                "document_types": self.metadata.get("document_types", []),
                "embedding_model": self.config.vector_embedding_model,
                "chunk_size": self.config.vector_chunk_size,
                "chunk_overlap": self.config.vector_chunk_overlap
            }
            
            # Get collection count
            try:
                collection_info["current_count"] = self.collection.count()
            except:
                collection_info["current_count"] = "Unknown"
            
            return collection_info
            
        except Exception as e:
            self.logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}
    
    async def clear_collection(self) -> Dict[str, Any]:
        """Clear all documents from the collection"""
        try:
            # Delete the collection
            self.client.delete_collection(self.collection_name)
            
            # Recreate the collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Attrition analysis documents and insights"}
            )
            
            # Reset metadata
            self.metadata = {
                "total_documents": 0,
                "last_updated": datetime.utcnow().isoformat(),
                "document_types": []
            }
            
            await self._save_metadata()
            
            self.logger.info("Collection cleared successfully")
            
            return {
                "success": True,
                "message": "Collection cleared successfully",
                "collection_name": self.collection_name
            }
            
        except Exception as e:
            self.logger.error(f"Failed to clear collection: {e}")
            raise
    
    async def export_collection(self, export_path: Optional[str] = None) -> str:
        """Export the collection data"""
        try:
            if export_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                export_path = f"./exports/vector_store_export_{timestamp}"
            
            # Create export directory
            Path(export_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Get all documents
            all_documents = self.collection.get()
            
            # Prepare export data
            export_data = {
                "collection_name": self.collection_name,
                "export_timestamp": datetime.utcnow().isoformat(),
                "total_documents": len(all_documents['ids']),
                "documents": []
            }
            
            # Add documents
            for i in range(len(all_documents['ids'])):
                doc_data = {
                    'id': all_documents['ids'][i],
                    'content': all_documents['documents'][i],
                    'metadata': all_documents['metadatas'][i] if all_documents['metadatas'] else {},
                    'embedding': all_documents['embeddings'][i] if all_documents['embeddings'] else None
                }
                export_data["documents"].append(doc_data)
            
            # Export to JSON
            json_path = f"{export_path}.json"
            with open(json_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            # Export to pickle for embeddings
            pickle_path = f"{export_path}.pkl"
            with open(pickle_path, 'wb') as f:
                pickle.dump(export_data, f)
            
            self.logger.info(f"Collection exported to {json_path} and {pickle_path}")
            
            return json_path
            
        except Exception as e:
            self.logger.error(f"Collection export failed: {e}")
            raise
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        try:
            return {
                "query_stats": self.query_stats,
                "index_stats": self.index_stats,
                "collection_info": await self.get_collection_info(),
                "embedding_model_info": {
                    "model_name": self.config.vector_embedding_model,
                    "max_sequence_length": getattr(self.embedding_model, 'max_seq_length', 'Unknown'),
                    "embedding_dimension": self.embedding_model.get_sentence_embedding_dimension()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get performance stats: {e}")
            return {"error": str(e)}
    
    async def _save_metadata(self):
        """Save metadata to file"""
        try:
            metadata_file = Path(self.config.vector_store_path) / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")
    
    async def shutdown(self):
        """Shutdown the vector store"""
        try:
            # Save metadata
            await self._save_metadata()
            
            # Clear references
            self.client = None
            self.collection = None
            self.embedding_model = None
            
            self.logger.info("Vector store shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Vector store shutdown failed: {e}")
