# Multi-Agent Attrition Analysis System

A comprehensive, AI-powered system for analyzing employee attrition using multiple specialized agents, LangChain, and Groq API.

## 🚀 Features

- **Multi-Agent Architecture**: Coordinated agents for data processing, analysis, prediction, insights, and chat
- **Agentic RAG**: Advanced retrieval-augmented generation for document analysis
- **Streamlit UI**: Modern, interactive web interface with real-time analytics
- **LangChain Integration**: Robust workflow orchestration and agent management
- **Groq API**: High-performance LLM inference for analysis and insights
- **Docker Support**: Containerized deployment with docker-compose

## 🏗️ Architecture

```
├── agents/                 # Multi-agent system
│   ├── base_agent.py      # Abstract base agent class
│   ├── coordinator_agent.py # Main orchestrator
│   ├── data_agent.py      # Data collection & processing
│   ├── analysis_agent.py  # Statistical analysis
│   ├── prediction_agent.py # ML predictions
│   ├── insight_agent.py   # Business insights
│   └── chat_agent.py      # RAG-powered chat
├── core/                   # Core system components
│   ├── config.py          # Configuration management
│   └── workflow.py        # Workflow orchestration
├── data/                   # Data handling
│   ├── data_loader.py     # Data collection
│   ├── data_processor.py  # Data processing
│   ├── feature_engineering.py # Feature engineering
│   └── schemas.py         # Data schemas
├── api/                    # FastAPI backend
│   └── main.py            # API endpoints
├── rag/                    # RAG components
│   ├── rag_agent.py       # RAG agent
│   └── vector_store.py    # Vector store management
├── utils/                  # Utilities
│   └── logger/            # Logging system
├── streamlit_integrated.py # Main Streamlit application
├── requirements.txt        # All dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker orchestration
└── env_template.env       # Environment template
```

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (optional)
- Groq API key

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd llm_pro
   ```

2. **Set up environment**
   ```bash
   cp env_template.env .env
   # Edit .env with your API keys and configuration
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run streamlit_integrated.py --server.port 8502
   ```

### Automated Installation

Use the provided installation script:

```bash
chmod +x install.sh
./install.sh
```

### Docker Deployment

```bash
docker-compose up -d
```

## 🔧 Configuration

### Environment Variables

- `GROQ_API_KEY`: Your Groq API key
- `DATABASE_URL`: PostgreSQL connection string
- `LOG_LEVEL`: Logging level (INFO, DEBUG, ERROR)
- `GROQ_MODEL`: LLM model name (default: llama3-8b-8192)

### API Keys

1. Get your Groq API key from [Groq Console](https://console.groq.com/)
2. Add it to your `.env` file
3. Restart the application

## 📱 Usage

### Streamlit Interface

1. **Dashboard**: Overview of system status and quick actions
2. **Analysis**: Data analysis and visualization
3. **Agents**: Agent management and monitoring
4. **Metrics**: Performance metrics and KPIs
5. **Chat**: AI-powered document chat with RAG
6. **Debug**: System health and diagnostics
7. **Settings**: Configuration management

### Demo Mode

The system includes a comprehensive demo mode that:
- Generates sample data and reports
- Runs analysis workflows
- Creates predictions and insights
- Demonstrates all agent capabilities
- Provides interactive visualizations

### Real System Mode

For production use:
1. Initialize the system from the Debug page
2. Upload your actual data
3. Configure agents and workflows
4. Run real analysis and predictions

## 🤖 Agents

### Coordinator Agent
- Orchestrates all other agents
- Manages workflows and job queues
- Handles inter-agent communication

### Data Agent
- Collects data from multiple sources
- Processes and cleans data
- Manages data pipelines

### Analysis Agent
- Performs statistical analysis
- Creates visualizations
- Generates insights reports

### Prediction Agent
- Builds ML models
- Makes predictions
- Evaluates model performance

### Insight Agent
- Generates business insights
- Identifies trends and patterns
- Provides actionable recommendations

### Chat Agent
- RAG-powered document chat
- Context-aware responses
- Document search and retrieval

## 🔍 API Endpoints

The system provides RESTful API endpoints for:
- Agent management
- Data operations
- Analysis workflows
- Prediction services
- Chat functionality

## 📊 Monitoring

- Real-time system health monitoring
- Agent status tracking
- Performance metrics
- Error logging and debugging

## 🚀 Deployment

### Production

1. Set production environment variables
2. Use Docker for containerization
3. Configure reverse proxy (nginx)
4. Set up monitoring and logging
5. Configure SSL certificates

### Scaling

- Horizontal scaling with multiple instances
- Load balancing
- Database clustering
- Cache optimization

## 🎯 Git Deployment

### Repository Setup

1. **Initialize Repository:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Multi-Agent Attrition Analysis System"
   ```

2. **Add Remote:**
   ```bash
   git remote add origin <your-repo-url>
   git branch -M main
   git push -u origin main
   ```

3. **Users Can Clone:**
   ```bash
   git clone <your-repo-url>
   cd llm_pro
   ./install.sh
   ```

### Repository Benefits

- **Clean Structure**: Easy to navigate and understand
- **No Bloat**: Only essential files included
- **Professional**: Ready for public/private deployment
- **Maintainable**: Clear separation of concerns
- **Documented**: Comprehensive setup instructions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the documentation
2. Review existing issues
3. Create a new issue with details
4. Contact the development team

## 🔄 Updates

- Regular updates and improvements
- Security patches
- New features and capabilities
- Performance optimizations

## 🔒 Security Notes

- `.env` file is in `.gitignore` (users create from template)
- No API keys or sensitive data included
- Docker configurations are secure
- All dependencies are from trusted sources

---

**Built with ❤️ using LangChain, Groq, and Streamlit**

---

## 🧹 Project Cleanup Summary

This repository has been cleaned and optimized for Git deployment:

### ❌ **Removed Files/Directories:**
- Test files, duplicate apps, unused entry points
- Sample data, cache directories, logs and exports
- Generated content, development notebooks
- Old environment files

### ✅ **Kept Files (Essential):**
- Complete Multi-Agent System (all 6 agents working)
- Streamlit Web Application (modern, interactive UI)
- RAG-Powered Chat (intelligent document analysis)
- Docker Support (containerized deployment)
- Comprehensive Documentation (setup and usage guides)
- Production Ready (clean, maintainable code)

**This repository is now Git-ready and contains everything needed for a professional, deployable multi-agent system! 🎉**
