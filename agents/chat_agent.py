"""
Chat Agent for Document Interaction and RAG-powered Conversations
"""
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import structlog

from .base_agent import BaseAgent, AgentMessage, AgentState
from data.schemas import ChatMessage, ChatResponse, DocumentQuery
from core.config import Config


@dataclass
class ChatContext:
    """Context for chat conversations"""
    session_id: str
    user_id: str
    conversation_history: List[ChatMessage]
    current_document: Optional[str] = None
    search_context: Optional[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class ChatAgent(BaseAgent):
    """
    Chat Agent for Document Interaction and RAG-powered Conversations
    
    Features:
    - Document Q&A using RAG
    - Conversational memory
    - Context-aware responses
    - Multi-document search
    - Chat session management
    """
    
    def __init__(self, agent_id: str = None):
        super().__init__(agent_id or f"ChatAgent_{self._generate_id()}")
        self.logger = structlog.get_logger(self.agent_id)
        
        # Chat-specific attributes
        self.active_sessions: Dict[str, ChatContext] = {}
        self.document_index: Dict[str, Any] = {}
        self.conversation_memory: Dict[str, List[ChatMessage]] = {}
        
        # RAG components
        self.embedding_model = None
        self.vector_store = None
        self.llm_chain = None
        
        self.logger.info("Chat agent initialized", agent_id=self.agent_id)
    
    async def initialize(self):
        """Initialize RAG components and document indexing"""
        try:
            # Initialize embedding model
            await self._setup_embeddings()
            
            # Initialize vector store
            await self._setup_vector_store()
            
            # Initialize LLM chain
            await self._setup_llm_chain()
            
            # Index existing documents
            await self._index_documents()
            
            self.logger.info("Chat agent RAG components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error("Failed to initialize chat agent", error=str(e))
            return False
    
    async def _initialize_agent(self):
        """Agent-specific initialization logic (required by BaseAgent)"""
        try:
            # Initialize RAG components and document indexing
            await self.initialize()
            return True
        except Exception as e:
            self.logger.error("Failed to initialize chat agent", error=str(e))
            return False
    
    async def _setup_embeddings(self):
        """Setup embedding model for document vectorization"""
        try:
            # Use sentence-transformers for embeddings
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.logger.info("Embedding model loaded successfully")
        except Exception as e:
            self.logger.error("Failed to load embedding model", error=str(e))
            raise
    
    async def _setup_vector_store(self):
        """Setup vector store for document search"""
        try:
            import chromadb
            self.vector_store = chromadb.Client()
            self.logger.info("Vector store initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize vector store", error=str(e))
            raise
    
    async def _setup_llm_chain(self):
        """Setup LLM chain for response generation"""
        try:
            from langchain_groq import ChatGroq
            from langchain.chains import ConversationalRetrievalChain
            from langchain.memory import ConversationBufferMemory
            from langchain.schema import BaseRetriever
            from langchain.schema import Document
            
            # Initialize Groq LLM
            llm = ChatGroq(
                groq_api_key=self.config.groq_api_key,
                model_name=self.config.groq_model
            )
            
            # Setup conversation memory
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # Create a simple retriever
            class SimpleRetriever(BaseRetriever):
                def __init__(self, documents):
                    self.documents = documents
                
                def _get_relevant_documents(self, query):
                    # Simple keyword matching for now
                    relevant_docs = []
                    query_lower = query.lower()
                    for doc in self.documents:
                        if any(word in doc['content'].lower() for word in query_lower.split()):
                            relevant_docs.append(Document(page_content=doc['content'], metadata=doc['metadata']))
                    return relevant_docs[:3]  # Return top 3 relevant documents
                
                async def aget_relevant_documents(self, query):
                    # Simple keyword matching for now
                    relevant_docs = []
                    query_lower = query.lower()
                    for doc in self.documents:
                        if any(word in doc['content'].lower() for word in query_lower.split()):
                            relevant_docs.append(Document(page_content=doc['content'], metadata=doc['metadata']))
                    return relevant_docs[:3]  # Return top 3 relevant documents
            
            # Create retriever with our sample documents
            documents_list = list(self.document_index.values())
            retriever = SimpleRetriever(documents_list)
            
            # Create conversational chain
            self.llm_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                return_source_documents=True
            )
            
            self.logger.info("LLM chain initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize LLM chain", error=str(e))
            raise
    
    async def _index_documents(self):
        """Index documents for RAG search"""
        try:
            # This would typically load and index documents from a database or file system
            # For now, we'll create a simple in-memory index
            sample_docs = [
                "Employee attrition is a critical HR metric that measures the rate at which employees leave an organization.",
                "High attrition rates can indicate poor job satisfaction, inadequate compensation, or lack of career growth opportunities.",
                "Data analysis shows that employees with 3-5 years of experience are most likely to leave for better opportunities.",
                "Companies with strong retention programs typically have 20-30% lower attrition rates than industry averages."
            ]
            
            for i, doc in enumerate(sample_docs):
                doc_id = f"doc_{i}"
                self.document_index[doc_id] = {
                    'content': doc,
                    'metadata': {'source': 'sample', 'id': doc_id}
                }
            
            self.logger.info(f"Indexed {len(sample_docs)} sample documents")
        except Exception as e:
            self.logger.error("Failed to index documents", error=str(e))
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """Process incoming chat messages"""
        try:
            if message.content_type == "chat_message":
                return await self._handle_chat_message(message)
            elif message.content_type == "document_query":
                return await self._handle_document_query(message)
            elif message.content_type == "session_management":
                return await self._handle_session_management(message)
            else:
                return self._create_error_response("Unsupported message type")
                
        except Exception as e:
            self.logger.error("Error processing message", error=str(e))
            return self._create_error_response(f"Processing error: {str(e)}")
    
    async def _handle_chat_message(self, message: AgentMessage) -> AgentMessage:
        """Handle regular chat messages"""
        try:
            chat_msg = ChatMessage(**message.content)
            
            # Get or create session
            session = self._get_or_create_session(chat_msg.session_id, chat_msg.user_id)
            
            # Add message to conversation history
            session.conversation_history.append(chat_msg)
            
            # Generate response using RAG
            response = await self._generate_rag_response(chat_msg.content, session)
            
            # Create response message
            chat_response = ChatResponse(
                session_id=chat_msg.session_id,
                user_id=chat_msg.user_id,
                message_id=f"resp_{self._generate_id()}",
                content=response['answer'],
                source_documents=response.get('sources', []),
                confidence=response.get('confidence', 0.8),
                timestamp=datetime.now()
            )
            
            # Add response to conversation history
            session.conversation_history.append(chat_response)
            
            return AgentMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                content_type="chat_response",
                content=chat_response.dict(),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error("Error handling chat message", error=str(e))
            return self._create_error_response(f"Chat error: {str(e)}")
    
    async def _handle_document_query(self, message: AgentMessage) -> AgentMessage:
        """Handle document-specific queries"""
        try:
            query = DocumentQuery(**message.content)
            
            # Search documents using RAG
            search_results = await self._search_documents(query.query, query.max_results or 5)
            
            # Generate response based on search results
            response = await self._generate_document_response(query.query, search_results)
            
            return AgentMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                content_type="document_response",
                content={
                    'query': query.query,
                    'response': response,
                    'sources': search_results,
                    'timestamp': datetime.now()
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error("Error handling document query", error=str(e))
            return self._create_error_response(f"Document query error: {str(e)}")
    
    async def _handle_session_management(self, message: AgentMessage) -> AgentMessage:
        """Handle session management requests"""
        try:
            action = message.content.get('action')
            session_id = message.content.get('session_id')
            
            if action == 'create':
                session = self._create_session(session_id, message.content.get('user_id', 'anonymous'))
                return self._create_success_response(f"Session created: {session_id}")
            
            elif action == 'delete':
                if session_id in self.active_sessions:
                    del self.active_sessions[session_id]
                    return self._create_success_response(f"Session deleted: {session_id}")
                else:
                    return self._create_error_response(f"Session not found: {session_id}")
            
            elif action == 'list':
                sessions = list(self.active_sessions.keys())
                return self._create_success_response(f"Active sessions: {sessions}")
            
            else:
                return self._create_error_response(f"Unknown action: {action}")
                
        except Exception as e:
            self.logger.error("Error handling session management", error=str(e))
            return self._create_error_response(f"Session management error: {str(e)}")
    
    async def _generate_rag_response(self, query: str, session: ChatContext) -> Dict[str, Any]:
        """Generate response using RAG (Retrieval Augmented Generation)"""
        try:
            if not self.llm_chain:
                return {
                    'answer': "I'm sorry, the AI system is not fully initialized yet. Please try again in a moment.",
                    'sources': [],
                    'confidence': 0.0
                }
            
            # Search for relevant documents
            search_results = await self._search_documents(query, max_results=3)
            
            # Create context from search results
            context = "\n".join([doc['content'] for doc in search_results])
            
            # Generate response using LLM
            response = await self.llm_chain.ainvoke({
                'question': query,
                'chat_history': [(msg.content, msg.response) for msg in session.conversation_history[-5:] if hasattr(msg, 'response')]
            })
            
            return {
                'answer': response['answer'],
                'sources': search_results,
                'confidence': 0.85  # Placeholder confidence score
            }
            
        except Exception as e:
            self.logger.error("Error generating RAG response", error=str(e))
            return {
                'answer': "I encountered an error while processing your request. Please try again.",
                'sources': [],
                'confidence': 0.0
            }
    
    async def _search_documents(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search documents using vector similarity"""
        try:
            if not self.embedding_model or not self.document_index:
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)
            
            # Simple similarity search (in production, use proper vector DB)
            results = []
            for doc_id, doc_info in self.document_index.items():
                doc_embedding = self.embedding_model.encode(doc_info['content'])
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                
                if similarity > 0.3:  # Threshold for relevance
                    results.append({
                        'id': doc_id,
                        'content': doc_info['content'],
                        'similarity': similarity,
                        'metadata': doc_info['metadata']
                    })
            
            # Sort by similarity and return top results
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:max_results]
            
        except Exception as e:
            self.logger.error("Error searching documents", error=str(e))
            return []
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        import numpy as np
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0
    
    async def _generate_document_response(self, query: str, search_results: List[Dict]) -> str:
        """Generate response based on document search results"""
        if not search_results:
            return "I couldn't find any relevant documents to answer your question."
        
        # Create a simple response based on search results
        context = "\n".join([doc['content'] for doc in search_results[:2]])
        
        return f"Based on the available documents, here's what I found:\n\n{context}\n\nThis information should help answer your question about '{query}'."
    
    def _get_or_create_session(self, session_id: str, user_id: str) -> ChatContext:
        """Get existing session or create new one"""
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = ChatContext(
                session_id=session_id,
                user_id=user_id,
                conversation_history=[]
            )
        return self.active_sessions[session_id]
    
    def _create_session(self, session_id: str, user_id: str) -> ChatContext:
        """Create a new chat session"""
        session = ChatContext(
            session_id=session_id,
            user_id=user_id,
            conversation_history=[]
        )
        self.active_sessions[session_id] = session
        return session
    
    def _create_success_response(self, message: str) -> AgentMessage:
        """Create a success response message"""
        return AgentMessage(
            sender_id=self.agent_id,
            recipient_id="system",
            content_type="success",
            content={'message': message, 'status': 'success'},
            timestamp=datetime.now()
        )
    
    def _create_error_response(self, message: str) -> AgentMessage:
        """Create an error response message"""
        return AgentMessage(
            sender_id=self.agent_id,
            recipient_id="system",
            content_type="error",
            content={'message': message, 'status': 'error'},
            timestamp=datetime.now()
        )
    
    async def get_chat_history(self, session_id: str) -> List[ChatMessage]:
        """Get chat history for a session"""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id].conversation_history
        return []
    
    async def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs"""
        return list(self.active_sessions.keys())
    
    async def cleanup_session(self, session_id: str) -> bool:
        """Clean up a chat session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            return True
        return False
