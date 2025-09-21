import os
import json
import sqlite3
import pickle
import hashlib
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_nltk():
    """Setup NLTK data with better error handling"""
    try:
        import nltk
        nltk_downloads = ['punkt', 'punkt_tab', 'stopwords', 'wordnet']
        
        for item in nltk_downloads:
            try:
                nltk.data.find(f'tokenizers/{item}')
                logger.info(f"✅ NLTK {item} already available")
            except LookupError:
                try:
                    logger.info(f"📥 Downloading NLTK {item}...")
                    nltk.download(item, quiet=True)
                    logger.info(f"✅ NLTK {item} downloaded successfully")
                except Exception as e:
                    logger.warning(f"⚠️ Could not download NLTK {item}: {e}")
        
        return True
    except ImportError:
        logger.warning("⚠️ NLTK not available")
        return False

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    NLTK_AVAILABLE = setup_nltk()
    
    if NLTK_AVAILABLE:
        import nltk
        from nltk.tokenize import sent_tokenize, word_tokenize
        from nltk.corpus import stopwords
        
        try:
            sent_tokenize("Test sentence.")
            word_tokenize("Test sentence.")
            TOKENIZERS_WORKING = True
        except:
            TOKENIZERS_WORKING = False
            logger.warning("⚠️ NLTK tokenizers not working, using fallback")
    else:
        TOKENIZERS_WORKING = False
    
    PACKAGES_AVAILABLE = True
    logger.info("✅ All core packages imported successfully")
    
except ImportError as e:
    logger.error(f"❌ Missing required packages: {e}")
    PACKAGES_AVAILABLE = False
    NLTK_AVAILABLE = False
    TOKENIZERS_WORKING = False

@dataclass
class Document:
    """Document data structure"""
    id: str
    title: str
    content: str
    source: str
    metadata: Dict[str, Any]
    timestamp: str
    embedding: Optional[np.ndarray] = None

@dataclass
class ResearchReport:
    """Research report structure"""
    query: str
    executive_summary: str
    key_findings: List[str]
    detailed_analysis: str
    sources: List[str]
    methodology: str
    confidence_score: float
    timestamp: str

class TextProcessor:
    """Handle text processing with fallbacks for NLTK issues"""
    
    def __init__(self):
        self.nltk_available = TOKENIZERS_WORKING
        if self.nltk_available:
            try:
                self.stop_words = set(stopwords.words('english'))
            except:
                self.stop_words = self.get_default_stopwords()
        else:
            self.stop_words = self.get_default_stopwords()
    
    def get_default_stopwords(self):
        """Fallback stopwords if NLTK not available"""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'under', 'over',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may',
            'might', 'must', 'shall', 'this', 'that', 'these', 'those', 'i', 'me',
            'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
            'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
            'theirs', 'themselves'
        }
    
    def sentence_tokenize(self, text: str) -> List[str]:
        """Tokenize text into sentences with fallback"""
        if self.nltk_available:
            try:
                return sent_tokenize(text)
            except:
                pass
        
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def word_tokenize(self, text: str) -> List[str]:
        """Tokenize text into words with fallback"""
        if self.nltk_available:
            try:
                return word_tokenize(text)
            except:
                pass
        
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:()-]', '', text)
        return text.strip()

class SimpleResearcherAgent:
    """Simplified Deep Researcher Agent with better error handling"""
    
    def __init__(self, data_directory: str = "data"):
        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(exist_ok=True)

        self.outputs_dir = Path("outputs")
        self.outputs_dir.mkdir(exist_ok=True)
        
        self.text_processor = TextProcessor()
        
        self.documents = []
        self.db_path = "research_agent.db"
        
        if PACKAGES_AVAILABLE:
            logger.info("Loading embedding model...")
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"❌ Error loading model: {e}")
                self.model = None
        else:
            self.model = None
        
        self.init_database()
        logger.info("🚀 Simple Researcher Agent initialized")

    def init_database(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    embedding BLOB
                )
            ''')
            conn.commit()

    def create_sample_documents(self):
        """Create sample documents for testing"""
        sample_dir = self.data_directory / "sample_documents"
        sample_dir.mkdir(exist_ok=True)
        
        samples = {
            "ai_overview.txt": """
Artificial Intelligence Overview

Artificial Intelligence (AI) refers to computer systems that can perform tasks typically requiring human intelligence. This includes learning, reasoning, problem-solving, perception, and language understanding.

Key AI Technologies:
- Machine Learning: Algorithms that improve through experience
- Deep Learning: Neural networks with multiple layers
- Natural Language Processing: Understanding and generating human language
- Computer Vision: Interpreting visual information
- Robotics: Physical AI applications

Current Applications:
- Healthcare: Medical diagnosis and drug discovery
- Finance: Fraud detection and algorithmic trading
- Transportation: Autonomous vehicles and traffic optimization
- Entertainment: Recommendation systems and content creation
- Education: Personalized learning and tutoring systems

Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. It uses statistical techniques to give computers the ability to learn patterns from data.

Types of Machine Learning:
- Supervised Learning: Learning from labeled examples
- Unsupervised Learning: Finding patterns in data without labels
- Reinforcement Learning: Learning through interaction and rewards

Future Prospects:
AI continues to advance rapidly, with potential applications in climate modeling, space exploration, and scientific research. Ethical considerations and responsible development remain crucial.
""",
            "quantum_computing.txt": """
Quantum Computing Fundamentals

Quantum computing harnesses quantum mechanical phenomena to process information in revolutionary ways. Unlike classical bits, quantum bits (qubits) can exist in superposition states.

Core Principles:
- Superposition: Qubits can be in multiple states simultaneously
- Entanglement: Quantum particles become correlated
- Quantum Interference: Amplifying correct solutions
- Quantum Gates: Operations on qubits

Key Advantages:
- Exponential speedup for certain problems
- Parallel processing capabilities
- Cryptographic applications
- Optimization problem solving

Classical Computing vs Quantum Computing:
Classical computers use bits that are either 0 or 1. Quantum computers use qubits that can be in superposition of both 0 and 1 simultaneously. This allows quantum computers to process exponentially more information than classical computers for certain types of problems.

Current Challenges:
- Quantum decoherence and error rates
- Limited number of stable qubits
- Need for extremely low temperatures
- Complex programming requirements

Applications:
- Cryptography and security
- Drug discovery and molecular simulation
- Financial modeling
- Weather prediction
- Artificial intelligence enhancement

Major players include IBM, Google, Microsoft, and various startups working on quantum hardware and software development.
""",
            "climate_science.txt": """
Climate Change Science

Climate change refers to long-term shifts in global temperatures and weather patterns. Scientific evidence overwhelmingly shows human activities as the primary driver since the mid-20th century.

Primary Causes:
- Greenhouse gas emissions from fossil fuels
- Deforestation and land use changes
- Industrial processes and manufacturing
- Agriculture and livestock farming
- Transportation systems

Observable Effects:
- Global temperature rise
- Melting ice caps and glaciers
- Sea level rise
- Extreme weather events
- Ecosystem disruption
- Ocean acidification

Scientific Consensus:
Over 97% of climate scientists agree that human activities are the primary cause of recent climate change. The evidence comes from multiple sources including temperature records, ice core data, and atmospheric measurements.

Mitigation Strategies:
- Renewable energy adoption
- Energy efficiency improvements
- Carbon capture and storage
- Sustainable transportation
- Forest conservation and reforestation
- International cooperation and policies

Adaptation Measures:
- Building climate-resilient infrastructure
- Developing drought-resistant crops
- Implementing early warning systems
- Protecting coastal areas from sea level rise

The Paris Climate Agreement aims to limit global warming to well below 2°C above pre-industrial levels, with efforts to limit it to 1.5°C.
"""
        }
        
        for filename, content in samples.items():
            file_path = sample_dir / filename
            if not file_path.exists():
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content.strip())
                logger.info(f"Created sample document: {filename}")
        
        return sample_dir

    def load_documents_from_directory(self, directory_path: str):
        """Load documents from directory with better error handling"""
        directory = Path(directory_path)
        if not directory.exists():
            logger.warning(f"Directory not found: {directory_path}")
            return
        
        txt_files = list(directory.glob("*.txt"))
        logger.info(f"Found {len(txt_files)} text files")
        
        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if not content:
                    logger.warning(f"Empty file: {file_path}")
                    continue
                
                content = self.text_processor.clean_text(content)
                
                doc_id = hashlib.md5(f"{file_path.name}{content[:100]}".encode()).hexdigest()
                
                document = Document(
                    id=doc_id,
                    title=file_path.stem.replace('_', ' ').title(),
                    content=content,
                    source=str(file_path),
                    metadata={
                        'file_type': '.txt', 
                        'word_count': len(content.split()),
                        'character_count': len(content)
                    },
                    timestamp=datetime.now().isoformat()
                )
                
                if self.model:
                    try:
                        document.embedding = self.model.encode([content])[0]
                    except Exception as e:
                        logger.warning(f"Could not generate embedding for {file_path}: {e}")
                
                self.store_document(document)
                self.documents.append(document)
                
                logger.info(f"✅ Loaded document: {document.title} ({document.metadata['word_count']} words)")
                
            except Exception as e:
                logger.error(f"❌ Error loading {file_path}: {e}")

    def store_document(self, document: Document):
        """Store document in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            embedding_blob = pickle.dumps(document.embedding) if document.embedding is not None else None
            
            cursor.execute('''
                INSERT OR REPLACE INTO documents 
                (id, title, content, source, metadata, timestamp, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                document.id, document.title, document.content, document.source,
                json.dumps(document.metadata), document.timestamp, embedding_blob
            ))
            conn.commit()

    def search_documents(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Search for relevant documents with fallback methods"""
        if not self.documents:
            logger.warning("No documents available for search")
            return []
        
        if self.model:
            try:
                return self.semantic_search(query, top_k)
            except Exception as e:
                logger.warning(f"Semantic search failed: {e}, falling back to keyword search")
        
        return self.keyword_search(query, top_k)
    
    def semantic_search(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        """Semantic search using embeddings"""
        query_embedding = self.model.encode([query])[0]
        
        results = []
        for doc in self.documents:
            if doc.embedding is not None:
                similarity = cosine_similarity([query_embedding], [doc.embedding])[0][0]
                results.append((doc, float(similarity)))
            else:
                score = self.calculate_keyword_similarity(query, doc.content)
                results.append((doc, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def keyword_search(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Fallback keyword-based search"""
        query_words = set(self.text_processor.word_tokenize(query.lower()))
        query_words = query_words - self.text_processor.stop_words
        
        results = []
        for doc in self.documents:
            score = self.calculate_keyword_similarity(query, doc.content)
            if score > 0:
                results.append((doc, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def calculate_keyword_similarity(self, query: str, content: str) -> float:
        """Calculate similarity based on keyword overlap"""
        query_words = set(self.text_processor.word_tokenize(query.lower()))
        content_words = set(self.text_processor.word_tokenize(content.lower()))
        query_words = query_words - self.text_processor.stop_words
        content_words = content_words - self.text_processor.stop_words
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words.intersection(content_words))
        
        query_coverage = overlap / len(query_words)
        content_relevance = overlap / max(len(content_words), 1)
        
        score = (query_coverage + content_relevance) / 2
        return score

    def research(self, query: str) -> ResearchReport:
        """Perform research on a query with better error handling"""
        logger.info(f"🔍 Researching: {query}")
        
        if not self.documents:
            return ResearchReport(
                query=query,
                executive_summary="No documents available for research. Please load some documents first.",
                key_findings=["No documents in the system"],
                detailed_analysis="The system has no documents to search through. Please add documents using load_documents_from_directory().",
                sources=[],
                methodology="No research performed due to lack of documents.",
                confidence_score=0.0,
                timestamp=datetime.now().isoformat()
            )
        
        try:
            search_results = self.search_documents(query, top_k=5)
            
            if not search_results:
                return ResearchReport(
                    query=query,
                    executive_summary="No relevant documents found for this query.",
                    key_findings=["No relevant information found"],
                    detailed_analysis="Unable to find relevant information in the document collection.",
                    sources=[],
                    methodology="Document search performed but no relevant content found.",
                    confidence_score=0.0,
                    timestamp=datetime.now().isoformat()
                )
            relevant_docs = [doc for doc, _ in search_results]
            key_findings = []
            all_sources = []
            
            for doc, similarity in search_results:
                sentences = self.text_processor.sentence_tokenize(doc.content)
                query_words = self.text_processor.word_tokenize(query.lower())
                query_words = [w for w in query_words if w not in self.text_processor.stop_words]
                
                relevant_sentences = []
                for sentence in sentences[:20]:
                    sentence_words = self.text_processor.word_tokenize(sentence.lower())
                    if any(word in sentence_words for word in query_words):
                        relevant_sentences.append(sentence.strip())
                
                if relevant_sentences:
                    finding = f"From '{doc.title}': {relevant_sentences[0][:200]}..."
                    key_findings.append(finding)
                elif similarity > 0.1: 
                    sentences = [s for s in sentences if len(s.split()) > 5]
                    if sentences:
                        finding = f"From '{doc.title}': {sentences[0][:200]}..."
                        key_findings.append(finding)
                
                all_sources.append(doc.source)

            executive_summary = f"Research on '{query}' analyzed {len(relevant_docs)} relevant documents. "
            if relevant_docs:
                doc_titles = [doc.title for doc in relevant_docs[:3]]
                executive_summary += f"Key information found from: {', '.join(doc_titles)}."
                if len(relevant_docs) > 3:
                    executive_summary += f" Plus {len(relevant_docs) - 3} additional sources."

            detailed_analysis = f"Detailed Analysis for: '{query}'\n\n"
            for i, (doc, similarity) in enumerate(search_results, 1):
                detailed_analysis += f"{i}. Source: {doc.title}\n"
                detailed_analysis += f"   Relevance Score: {similarity:.3f}\n"
                detailed_analysis += f"   Content Preview: {doc.content[:300]}...\n\n"
            
            if search_results:
                avg_similarity = sum(sim for _, sim in search_results) / len(search_results)
                confidence_score = min(avg_similarity * 1.5, 1.0) 
            else:
                confidence_score = 0.0
            
            methodology = f"Performed {'semantic' if self.model else 'keyword-based'} search across {len(self.documents)} documents. "
            methodology += f"Found {len(search_results)} relevant sources with average relevance of {avg_similarity:.3f}."
            
            report = ResearchReport(
                query=query,
                executive_summary=executive_summary,
                key_findings=key_findings[:5],  
                detailed_analysis=detailed_analysis,
                sources=list(set(all_sources)),
                methodology=methodology,
                confidence_score=confidence_score,
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"✅ Research completed with {len(key_findings)} key findings")
            return report
            
        except Exception as e:
            logger.error(f"❌ Error during research: {e}")
            return ResearchReport(
                query=query,
                executive_summary=f"Error occurred during research: {str(e)}",
                key_findings=[f"Error: {str(e)}"],
                detailed_analysis=f"An error occurred while processing the query: {str(e)}",
                sources=[],
                methodology="Research failed due to error.",
                confidence_score=0.0,
                timestamp=datetime.now().isoformat()
            )

    def export_report_to_markdown(self, report: ResearchReport, filename: str = None):
        """Export research report to Markdown"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"research_report_{timestamp}.md"
        
        filepath = self.outputs_dir / filename
        
        md_content = f"""# Research Report: {report.query}

**Generated:** {report.timestamp}  
**Confidence Score:** {report.confidence_score:.2f}

## Executive Summary

{report.executive_summary}

## Key Findings

"""
        
        for i, finding in enumerate(report.key_findings, 1):
            md_content += f"{i}. {finding}\n\n"
        
        md_content += f"""

## Detailed Analysis

{report.detailed_analysis}

## Methodology

{report.methodology}

## Sources

"""
        
        for i, source in enumerate(report.sources, 1):
            md_content += f"{i}. {source}\n"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"📄 Report exported to {filepath}")
        return str(filepath)

    def interactive_research(self):
        """Interactive research session"""
        print("\n🔬 Deep Researcher Agent - Interactive Mode")
        print("=" * 50)
        print("Type 'quit' or 'exit' to end the session")
        print("Type 'stats' to see system statistics")
        print("Type 'help' for commands\n")
        
        while True:
            query = input("🤔 Enter your research question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Thank you for using Deep Researcher Agent!")
                break
            
            if query.lower() == 'stats':
                stats = self.get_stats()
                print(f"\n📊 SYSTEM STATISTICS:")
                for key, value in stats.items():
                    print(f"   {key.replace('_', ' ').title()}: {value}")
                continue
            
            if query.lower() == 'help':
                print(f"\n📚 AVAILABLE COMMANDS:")
                print("   - Ask any research question")
                print("   - 'stats' - Show system statistics")
                print("   - 'quit' or 'exit' - End session")
                print("   - 'help' - Show this help")
                continue
            
            if not query:
                continue
            
            print(f"\n🔍 Researching: {query}")
            print("-" * 40)
            
            report = self.research(query)
            
            print(f"\n📋 EXECUTIVE SUMMARY:")
            print(report.executive_summary)
            
            print(f"\n🔍 KEY FINDINGS:")
            for i, finding in enumerate(report.key_findings, 1):
                print(f"{i}. {finding}")
            
            print(f"\n📊 CONFIDENCE SCORE: {report.confidence_score:.2f}")
            print(f"📚 SOURCES: {len(report.sources)} documents analyzed")
            
            export_choice = input("\n💾 Export this report? (y/n): ").strip().lower()
            if export_choice in ['y', 'yes']:
                filename = self.export_report_to_markdown(report)
                print(f"📄 Report saved to: {filename}")
            
            print("\n" + "="*50)

    def get_stats(self):
        """Get system statistics"""
        total_words = sum(doc.metadata.get('word_count', 0) for doc in self.documents)
        total_chars = sum(doc.metadata.get('character_count', 0) for doc in self.documents)
        
        stats = {
            'total_documents': len(self.documents),
            'total_words': f"{total_words:,}",
            'total_characters': f"{total_chars:,}",
            'ai_model_status': '✅ Active' if self.model else '❌ Not available',
            'nltk_status': '✅ Working' if TOKENIZERS_WORKING else '⚠️ Limited',
            'database_path': self.db_path,
            'outputs_directory': str(self.outputs_dir)
        }
        return stats

def main():
    """Main function to demonstrate the system"""
    print("🚀 Starting Deep Researcher Agent (Fixed Version)...")
    print("=" * 60)
    
    agent = SimpleResearcherAgent()
    
    print("\n📁 Creating sample documents...")
    sample_dir = agent.create_sample_documents()
    
    print("📚 Loading documents...")
    agent.load_documents_from_directory(str(sample_dir))
    
    stats = agent.get_stats()
    print(f"\n📊 SYSTEM STATUS:")
    for key, value in stats.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    if len(agent.documents) > 0:
        print(f"\n🔬 Running example research queries...")
        
        example_queries = [
            "What is artificial intelligence?",
            "How does quantum computing work?",
            "What causes climate change?"
        ]
        
        for i, query in enumerate(example_queries, 1):
            print(f"\n{'='*50}")
            print(f"🔍 QUERY {i}: {query}")
            print("="*50)
            
            try:
                report = agent.research(query)
                
                print(f"\n📋 SUMMARY:")
                print(f"   {report.executive_summary}")
                
                print(f"\n📊 CONFIDENCE: {report.confidence_score:.2f}")
                
                if report.key_findings:
                    print(f"\n🔍 TOP FINDINGS:")
                    for finding in report.key_findings[:2]:
                        print(f"   • {finding}")
                
                if i == 1:
                    filename = agent.export_report_to_markdown(report, "example_report.md")
                    print(f"\n💾 Sample report saved: {filename}")
                    
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
    else:
        print("\n⚠️ No documents loaded. Please check the sample_documents directory.")
    
    print(f"\n✅ Setup complete! System is ready to use.")
    
    start_interactive = input("\n🤔 Start interactive research mode? (y/n): ").strip().lower()
    if start_interactive in ['y', 'yes']:
        agent.interactive_research()
    
    return agent

if __name__ == "__main__":
    try:
        agent = main()
        
        print(f"\n" + "="*60)
        print("🎯 SYSTEM READY - USAGE GUIDE")
        print("="*60)
        print("""
# Basic usage:
agent = SimpleResearcherAgent()
agent.create_sample_documents()
agent.load_documents_from_directory("data/sample_documents")

# Research a topic:
report = agent.research("your question here")
print(report.executive_summary)

# Export results:
agent.export_report_to_markdown(report)

# Interactive mode:
agent.interactive_research()

# Check system status:
print(agent.get_stats())

# Add your own documents:
# 1. Create .txt files in the data/ directory
# 2. Run: agent.load_documents_from_directory("data")
# 3. Start researching!
""")
    
    except Exception as e:
        print(f"\n❌ Startup error: {e}")
        print("Please check that all required packages are installed:")
        print("pip install sentence-transformers scikit-learn nltk numpy pandas")
        print("\nIf NLTK issues persist, run:")
        print("python3 -c \"import nltk; nltk.download('punkt_tab'); nltk.download('punkt'); nltk.download('stopwords')\"")
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")