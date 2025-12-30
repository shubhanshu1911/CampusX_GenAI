# GenAI with Langchain - Learning Repository

A comprehensive learning repository covering Langchain fundamentals and advanced concepts for building Generative AI applications.

### Full Detailed Notes : [Campus X GenAI Notes](https://docs.google.com/document/d/11Y8XxdfRC92e-N5crHJKid0ICuGAI8EQ25oIOnXyTso/edit?usp=sharing)

## 📚 Overview

This repository contains hands-on examples and implementations covering the complete Langchain ecosystem, from basic models and prompts to advanced RAG systems and AI agents.

## 🏗️ Project Structure

The repository is organized into 12 chapters, each focusing on specific Langchain concepts:

- **Ch1_Langchain_models/** - Chat models (OpenAI, Gemini) and embedding models
- **Ch2_Langchain_Prompts/** - Prompt templates, message placeholders, and UI examples
- **Ch3_Langchain_structured_output/** - Structured output parsing (Pydantic, JSON, TypedDict)
- **Ch4_Langchian_chains/** - Sequential, parallel, and conditional chains
- **Ch5_Langchain_Runnables/** - Runnable interfaces, branches, and lambda functions
- **Ch6_Document_loaders/** - Loading documents from various sources (PDF, CSV, web, text)
- **Ch7_Text_splitter/** - Text splitting strategies (length-based, structure-based, document-based)
- **Ch8_Vector_DB/** - Vector database operations with ChromaDB
- **Ch9_Retrivals/** - Document retrieval techniques
- **Ch10_Yt_chatbot/** - YouTube RAG chatbot with LangSmith evaluation
- **Ch11_Tools/** - Tool creation, calling, and integration
- **Ch12_AI_Agent/** - Building AI agents with ReAct framework

## 🚀 Setup

### Prerequisites

- Python 3.10+
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd "GenAI with Langchain"
```

2. Create and activate a virtual environment:
```bash
python -m venv VENV
source VENV/bin/activate  # On Windows: VENV\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your API keys:
```env
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
LANGCHAIN_API_KEY=your_langsmith_api_key  # Optional, for LangSmith tracing
```

## 📦 Dependencies

- **LangChain Core**: `langchain`, `langchain-core`
- **Model Integrations**: `langchain-openai`, `langchain-anthropic`, `langchain-google-genai`, `langchain-huggingface`
- **Utilities**: `python-dotenv`, `numpy`, `scikit-learn`

See `requirements.txt` for the complete list.

## 💡 Usage Examples

### Basic Chat Model
```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=0.7)
result = model.invoke("Hello, how are you?")
print(result.content)
```

### Simple Chain
```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

chain = prompt | model | StrOutputParser()
result = chain.invoke({'topic': 'cricket'})
```

## 🎯 Key Features

- **Multi-Provider Support**: Examples with OpenAI, Anthropic, Google Gemini, and Hugging Face
- **RAG Implementation**: Complete RAG pipeline with YouTube transcript processing
- **Agent Development**: ReAct agents with tool integration
- **Evaluation**: LangSmith integration for monitoring and evaluation
- **UI Examples**: Streamlit-based prompt UI demonstrations

## 📝 Notes

- Most examples use Google Gemini (`gemini-2.5-flash-preview-05-20`) as the default model
- Jupyter notebooks are available for interactive learning in several chapters
- The project includes sample data files (PDFs, text files) for testing document loaders
- Vector database examples use ChromaDB (local storage in `Ch8_Vector_DB/my_chroma_db/`)

## 🔗 Resources

- [Langchain Documentation](https://python.langchain.com/)
- [LangSmith](https://smith.langchain.com/) - For tracing and evaluation
- [Langchain Hub](https://smith.langchain.com/hub) - For prompt templates

## 📄 License

This is a learning repository for educational purposes.

