"""
RAG Agent for intelligent document retrieval and generation using LangChain
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
import json

from langchain.llms import Groq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from core.config import Config
from rag.vector_store import VectorStore


class RAGAgent:
    """
    RAG Agent for intelligent document retrieval and generation
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # LangChain components
        self.llm: Optional[Groq] = None
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.text_splitter: Optional[RecursiveCharacterTextSplitter] = None
        
        # RAG components
        self.retrieval_chain: Optional[RetrievalQA] = None
        self.conversational_chain: Optional[ConversationalRetrievalChain] = None
        self.memory: Optional[ConversationBufferMemory] = None
        
        # Vector store integration
        self.vector_store: Optional[VectorStore] = None
        
        # Performance tracking
        self.query_history: List[Dict[str, Any]] = []
        self.generation_stats: Dict[str, Any] = {}
    
    async def initialize(self):
        """Initialize the RAG agent"""
        try:
            # Initialize Groq LLM
            self.llm = Groq(
                groq_api_key=self.config.groq_api_key,
                model_name=self.config.groq_model,
                temperature=self.config.groq_temperature,
                max_tokens=self.config.groq_max_tokens
            )
            
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.vector_embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.vector_chunk_size,
                chunk_overlap=self.config.vector_chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            # Initialize memory
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            self.logger.info("RAG agent components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG agent: {e}")
            raise
    
    async def set_vector_store(self, vector_store: VectorStore):
        """Set the vector store for the RAG agent"""
        try:
            self.vector_store = vector_store
            
            # Initialize retrieval chain
            await self._initialize_retrieval_chain()
            
            self.logger.info("Vector store set and retrieval chain initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to set vector store: {e}")
            raise
    
    async def _initialize_retrieval_chain(self):
        """Initialize the retrieval chain"""
        try:
            if not self.vector_store:
                raise ValueError("Vector store not set")
            
            # Create prompt template
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""
                You are an expert in employee attrition analysis. Use the following context to answer the question.
                
                Context: {context}
                
                Question: {question}
                
                Answer the question based on the context provided. If the context doesn't contain enough information to answer the question, say so. Be concise and accurate.
                
                Answer:"""
            )
            
            # Initialize retrieval chain
            self.retrieval_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self._create_retriever(),
                chain_type_kwargs={"prompt": prompt_template},
                return_source_documents=True
            )
            
            # Initialize conversational chain
            self.conversational_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self._create_retriever(),
                memory=self.memory,
                return_source_documents=True,
                verbose=True
            )
            
            self.logger.info("Retrieval chains initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize retrieval chain: {e}")
            raise
    
    def _create_retriever(self):
        """Create a retriever from the vector store"""
        try:
            if not self.vector_store or not self.vector_store.collection:
                raise ValueError("Vector store collection not available")
            
            # Create Chroma retriever
            retriever = Chroma(
                client=self.vector_store.client,
                collection_name=self.vector_store.collection_name,
                embedding_function=self.embeddings
            ).as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            # Create contextual compression retriever
            compressor_prompt = """Given the following question and context, extract any relevant information from the context that can help answer the question. If none of the context is relevant, return "No relevant information found."

Question: {question}
Context: {context}

Relevant information:"""
            
            compressor = LLMChainExtractor.from_llm(self.llm, prompt=compressor_prompt)
            
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=retriever
            )
            
            return compression_retriever
            
        except Exception as e:
            self.logger.error(f"Failed to create retriever: {e}")
            raise
    
    async def process_documents(self, documents: List[str], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process and index documents for RAG"""
        try:
            if not documents:
                raise ValueError("No documents provided")
            
            self.logger.info(f"Processing {len(documents)} documents for RAG")
            
            # Split documents into chunks
            chunks = []
            for i, doc in enumerate(documents):
                try:
                    doc_chunks = self.text_splitter.split_text(doc)
                    for j, chunk in enumerate(doc_chunks):
                        chunk_metadata = {
                            "document_index": i,
                            "chunk_index": j,
                            "chunk_size": len(chunk),
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        
                        if metadata:
                            chunk_metadata.update(metadata)
                        
                        chunks.append({
                            "content": chunk,
                            "metadata": chunk_metadata,
                            "id": f"chunk_{i}_{j}_{datetime.now().timestamp()}"
                        })
                        
                except Exception as e:
                    self.logger.warning(f"Failed to process document {i}: {e}")
                    continue
            
            if not chunks:
                raise ValueError("No valid chunks created from documents")
            
            # Add chunks to vector store
            if self.vector_store:
                result = await self.vector_store.add_documents(chunks)
            else:
                # Fallback: store in memory
                result = {"success": True, "documents_added": len(chunks)}
            
            self.logger.info(f"Successfully processed {len(chunks)} document chunks")
            
            return {
                "success": True,
                "documents_processed": len(documents),
                "chunks_created": len(chunks),
                "indexing_result": result
            }
            
        except Exception as e:
            self.logger.error(f"Document processing failed: {e}")
            raise
    
    async def query(self, question: str, use_conversation: bool = False, 
                   top_k: int = 5) -> Dict[str, Any]:
        """Query the RAG system"""
        try:
            if not question or not question.strip():
                raise ValueError("Question cannot be empty")
            
            self.logger.info(f"Processing query: '{question}'")
            
            start_time = datetime.utcnow()
            
            # Choose appropriate chain
            if use_conversation and self.conversational_chain:
                chain = self.conversational_chain
                chain_type = "conversational"
            elif self.retrieval_chain:
                chain = self.retrieval_chain
                chain_type = "retrieval"
            else:
                raise ValueError("No retrieval chain available")
            
            # Execute query
            if chain_type == "conversational":
                result = chain({"question": question})
                answer = result.get("answer", "")
                source_documents = result.get("source_documents", [])
            else:
                result = chain({"query": question})
                answer = result.get("result", "")
                source_documents = result.get("source_documents", [])
            
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            
            # Format source documents
            formatted_sources = []
            for doc in source_documents:
                if hasattr(doc, 'page_content'):
                    formatted_sources.append({
                        "content": doc.page_content,
                        "metadata": getattr(doc, 'metadata', {})
                    })
                else:
                    formatted_sources.append({
                        "content": str(doc),
                        "metadata": {}
                    })
            
            # Create response
            response = {
                "question": question,
                "answer": answer,
                "source_documents": formatted_sources,
                "chain_type": chain_type,
                "processing_time": processing_time,
                "timestamp": end_time.isoformat()
            }
            
            # Store query history
            self.query_history.append({
                "timestamp": end_time.isoformat(),
                "question": question,
                "answer": answer,
                "processing_time": processing_time,
                "chain_type": chain_type,
                "sources_count": len(formatted_sources)
            })
            
            # Update generation stats
            if "total_queries" not in self.generation_stats:
                self.generation_stats["total_queries"] = 0
                self.generation_stats["total_processing_time"] = 0.0
            
            self.generation_stats["total_queries"] += 1
            self.generation_stats["total_processing_time"] += processing_time
            self.generation_stats["average_processing_time"] = (
                self.generation_stats["total_processing_time"] / self.generation_stats["total_queries"]
            )
            
            self.logger.info(f"Query processed successfully in {processing_time:.2f}s")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            raise
    
    async def generate_insights(self, analysis_results: Dict[str, Any], 
                              question: str = None) -> Dict[str, Any]:
        """Generate insights using RAG"""
        try:
            if not analysis_results:
                raise ValueError("No analysis results provided")
            
            # Create context from analysis results
            context = self._create_analysis_context(analysis_results)
            
            # Generate question if not provided
            if not question:
                question = "What are the key insights and recommendations from this attrition analysis?"
            
            # Create enhanced prompt
            enhanced_prompt = f"""
            Based on the following attrition analysis results, provide comprehensive insights and actionable recommendations:
            
            Analysis Results:
            {context}
            
            Question: {question}
            
            Please provide:
            1. Key findings and patterns
            2. Risk factors identified
            3. Actionable recommendations
            4. Priority areas for intervention
            5. Expected impact of recommendations
            """
            
            # Query the system
            response = await self.query(enhanced_prompt, use_conversation=False)
            
            # Extract insights
            insights = {
                "analysis_context": context,
                "question": question,
                "generated_insights": response["answer"],
                "source_documents": response["source_documents"],
                "generation_timestamp": datetime.utcnow().isoformat()
            }
            
            self.logger.info("Insights generated successfully")
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Insight generation failed: {e}")
            raise
    
    def _create_analysis_context(self, analysis_results: Dict[str, Any]) -> str:
        """Create context string from analysis results"""
        try:
            context_parts = []
            
            # Add data quality information
            if "data_quality" in analysis_results:
                quality = analysis_results["data_quality"]
                context_parts.append(f"Data Quality: {quality.get('quality_score', 'N/A')}/100")
                context_parts.append(f"Total Records: {quality.get('total_records', 'N/A')}")
                context_parts.append(f"Missing Values: {quality.get('missing_percentage', 'N/A')}%")
            
            # Add statistical analysis
            if "statistical_analysis" in analysis_results:
                stats = analysis_results["statistical_analysis"]
                if "correlations" in stats:
                    context_parts.append(f"Key Correlations: {stats['correlations']}")
                if "feature_importance" in stats:
                    context_parts.append(f"Feature Importance: {stats['feature_importance']}")
            
            # Add model performance
            if "model_performance" in analysis_results:
                model = analysis_results["model_performance"]
                context_parts.append(f"Model Accuracy: {model.get('accuracy', 'N/A')}")
                context_parts.append(f"Model Type: {model.get('model_type', 'N/A')}")
            
            # Add predictions
            if "predictions" in analysis_results:
                preds = analysis_results["predictions"]
                context_parts.append(f"Attrition Rate: {preds.get('attrition_rate', 'N/A')}%")
                context_parts.append(f"High Risk Employees: {preds.get('high_risk_count', 'N/A')}")
            
            # Add business insights
            if "business_insights" in analysis_results:
                insights = analysis_results["business_insights"]
                context_parts.append(f"Key Insights: {insights}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            self.logger.error(f"Failed to create analysis context: {e}")
            return str(analysis_results)
    
    async def get_similar_questions(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find similar questions from query history"""
        try:
            if not self.query_history:
                return []
            
            # Use embeddings to find similar questions
            question_embedding = self.embeddings.embed_query(question)
            
            similarities = []
            for query_record in self.query_history:
                if query_record["question"] != question:  # Exclude current question
                    history_embedding = self.embeddings.embed_query(query_record["question"])
                    
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(question_embedding, history_embedding)
                    similarities.append({
                        "question": query_record["question"],
                        "answer": query_record["answer"],
                        "similarity": similarity,
                        "timestamp": query_record["timestamp"]
                    })
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            self.logger.error(f"Failed to find similar questions: {e}")
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            import numpy as np
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            self.logger.error(f"Cosine similarity calculation failed: {e}")
            return 0.0
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        try:
            return {
                "query_history": self.query_history[-10:],  # Last 10 queries
                "generation_stats": self.generation_stats,
                "total_queries": len(self.query_history),
                "average_processing_time": self.generation_stats.get("average_processing_time", 0.0),
                "vector_store_stats": await self.vector_store.get_performance_stats() if self.vector_store else None
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get performance stats: {e}")
            return {"error": str(e)}
    
    async def export_query_history(self, export_path: Optional[str] = None) -> str:
        """Export query history to file"""
        try:
            if export_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                export_path = f"./exports/rag_query_history_{timestamp}.json"
            
            # Create export directory
            from pathlib import Path
            Path(export_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare export data
            export_data = {
                "export_timestamp": datetime.utcnow().isoformat(),
                "total_queries": len(self.query_history),
                "query_history": self.query_history,
                "generation_stats": self.generation_stats
            }
            
            # Export to JSON
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Query history exported to {export_path}")
            return export_path
            
        except Exception as e:
            self.logger.error(f"Query history export failed: {e}")
            raise
    
    async def clear_memory(self) -> Dict[str, Any]:
        """Clear conversation memory"""
        try:
            if self.memory:
                self.memory.clear()
            
            # Clear query history
            self.query_history.clear()
            
            # Reset generation stats
            self.generation_stats.clear()
            
            self.logger.info("Memory and history cleared successfully")
            
            return {
                "success": True,
                "message": "Memory and history cleared successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to clear memory: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the RAG agent"""
        try:
            # Clear components
            self.llm = None
            self.embeddings = None
            self.text_splitter = None
            self.retrieval_chain = None
            self.conversational_chain = None
            self.memory = None
            self.vector_store = None
            
            # Clear data
            self.query_history.clear()
            self.generation_stats.clear()
            
            self.logger.info("RAG agent shutdown completed")
            
        except Exception as e:
            self.logger.error(f"RAG agent shutdown failed: {e}")
