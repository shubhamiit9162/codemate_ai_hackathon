from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global agent instances
research_agent = None
super_agent = None

def initialize_agents():
    """Initialize research agents with proper error handling"""
    global research_agent, super_agent
    
    try:
        # Try to import your research agents
        from index import SimpleResearcherAgent
        logger.info("Successfully imported SimpleResearcherAgent")
        
        # Initialize standard agent
        research_agent = SimpleResearcherAgent()
        
        # Create and load sample documents
        sample_dir = research_agent.create_sample_documents()
        research_agent.load_documents_from_directory(str(sample_dir))
        logger.info("Standard research agent initialized successfully")
        
        # Try to initialize super agent
        try:
            from super_confidence_agent import SuperConfidenceAgent
            super_agent = SuperConfidenceAgent()
            super_agent.load_documents_from_directory(str(sample_dir))
            logger.info("Super confidence agent initialized successfully")
        except ImportError as e:
            logger.warning(f"Super confidence agent not available: {e}")
            super_agent = None
            
    except Exception as e:
        logger.error(f"Error initializing agents: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Create fallback agent if main agents fail
        research_agent = create_fallback_agent()
        super_agent = research_agent
        logger.info("Using fallback agent due to initialization errors")

def create_fallback_agent():
    """Create a simple fallback agent if main agents fail"""
    
    class FallbackAgent:
        def __init__(self):
            self.documents = [
                {
                    'id': '1',
                    'title': 'Artificial Intelligence Overview',
                    'content': """Artificial Intelligence (AI) refers to computer systems that can perform tasks typically requiring human intelligence. This includes learning, reasoning, problem-solving, perception, and language understanding. Key AI technologies include Machine Learning, Deep Learning, Natural Language Processing, Computer Vision, and Robotics. Current applications span healthcare, finance, transportation, entertainment, and education. Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed."""
                },
                {
                    'id': '2',
                    'title': 'Quantum Computing Fundamentals', 
                    'content': """Quantum computing harnesses quantum mechanical phenomena to process information in revolutionary ways. Unlike classical bits, quantum bits (qubits) can exist in superposition states. Core principles include superposition, entanglement, quantum interference, and quantum gates. This allows exponential speedup for certain problems like cryptography, optimization, and molecular simulation. Classical computers use bits that are either 0 or 1, while quantum computers use qubits that can be in superposition of both states simultaneously."""
                },
                {
                    'id': '3',
                    'title': 'Climate Change Science',
                    'content': """Climate change refers to long-term shifts in global temperatures and weather patterns. Human activities are the primary driver since the mid-20th century, mainly through greenhouse gas emissions from fossil fuels, deforestation, and industrial processes. Observable effects include global temperature rise, melting ice caps, sea level rise, and extreme weather events. Over 97% of climate scientists agree that human activities are the primary cause of recent climate change."""
                }
            ]
            
            # Create outputs directory
            self.outputs_dir = Path("outputs")
            self.outputs_dir.mkdir(exist_ok=True)
        
        def search_documents(self, query, top_k=5):
            """Simple keyword-based document search"""
            query_words = set(query.lower().split())
            results = []
            
            for doc in self.documents:
                content_words = set(doc['content'].lower().split())
                title_words = set(doc['title'].lower().split())
                
                content_matches = len(query_words.intersection(content_words))
                title_matches = len(query_words.intersection(title_words)) * 2
                
                if content_matches > 0 or title_matches > 0:
                    score = (content_matches + title_matches) / len(query_words)
                    results.append((doc, score))
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
        
        def research(self, query):
            """Perform research and generate report"""
            search_results = self.search_documents(query)
            
            findings = []
            sources = []
            
            for doc, score in search_results:
                findings.append(f"From '{doc['title']}': {doc['content'][:200]}...")
                sources.append(doc['title'])
            
            confidence = min(len(search_results) * 0.4, 1.0) if search_results else 0.0
            
            # Create report object that matches expected structure
            class Report:
                def __init__(self):
                    self.query = query
                    self.executive_summary = f'Research on "{query}" found {len(search_results)} relevant documents with useful information.'
                    self.key_findings = findings
                    self.detailed_analysis = f'Analyzed {len(search_results)} documents using keyword matching algorithm. Found relevant content in documents covering the requested topic.'
                    self.sources = sources
                    self.methodology = 'Simple keyword-based search with title weighting and content analysis'
                    self.confidence_score = confidence
                    self.timestamp = datetime.now().isoformat()
            
            return Report()
        
        def get_stats(self):
            """Get system statistics"""
            return {
                'total_documents': len(self.documents),
                'total_words': f"{sum(len(doc['content'].split()) for doc in self.documents):,}",
                'total_characters': f"{sum(len(doc['content']) for doc in self.documents):,}",
                'ai_model_status': 'Fallback keyword search',
                'nltk_status': 'Not required for fallback mode',
                'database_path': 'In-memory storage',
                'outputs_directory': str(self.outputs_dir)
            }
        
        def export_report_to_markdown(self, report, filename=None):
            """Export report to markdown file"""
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
            
            return str(filepath)
        
        def load_documents_from_directory(self, directory):
            """Stub for compatibility"""
            logger.info(f"Fallback agent: ignoring directory load request for {directory}")
            pass
    
    return FallbackAgent()

@app.route('/')
def index():
    """Main page"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error serving index page: {e}")
        return f"<h1>Deep Research Agent</h1><p>Error loading template: {e}</p>", 500

@app.route('/api/research', methods=['POST'])
def research():
    """Perform research on a query"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        query = data.get('query', '').strip()
        agent_type = data.get('agent_type', 'standard')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        logger.info(f"Processing research query: '{query}' with agent type: {agent_type}")
        
        # Select agent
        if agent_type == 'super' and super_agent:
            agent = super_agent
        else:
            agent = research_agent
        
        if not agent:
            return jsonify({'error': 'Research agent not available'}), 500
        
        # Perform research
        report = agent.research(query)
        
        # Convert to dict for JSON serialization
        response = {
            'query': report.query,
            'executive_summary': report.executive_summary,
            'key_findings': report.key_findings,
            'detailed_analysis': report.detailed_analysis,
            'sources': report.sources,
            'methodology': report.methodology,
            'confidence_score': report.confidence_score,
            'timestamp': report.timestamp,
            'agent_type': agent_type
        }
        
        logger.info(f"Research completed successfully with confidence: {report.confidence_score:.3f}")
        return jsonify(response)
        
    except Exception as e:
        error_msg = f"Research error: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        agent = research_agent
        if not agent:
            return jsonify({'error': 'No agent available'}), 500
            
        stats = agent.get_stats()
        logger.info("Stats retrieved successfully")
        return jsonify(stats)
        
    except Exception as e:
        error_msg = f"Stats error: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500

@app.route('/api/export', methods=['POST'])
def export_report():
    """Export research report"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        report_data = data.get('report')
        
        if not report_data:
            return jsonify({'error': 'Report data required'}), 400
        
        # Create a temporary report object
        class TempReport:
            def __init__(self, data):
                self.query = data['query']
                self.executive_summary = data['executive_summary']
                self.key_findings = data['key_findings']
                self.detailed_analysis = data['detailed_analysis']
                self.sources = data['sources']
                self.methodology = data['methodology']
                self.confidence_score = data['confidence_score']
                self.timestamp = data['timestamp']
        
        temp_report = TempReport(report_data)
        
        # Use available agent to export
        agent = research_agent
        if not agent:
            return jsonify({'error': 'No agent available'}), 500
            
        filename = agent.export_report_to_markdown(temp_report)
        logger.info(f"Report exported to: {filename}")
        
        return jsonify({'filename': filename, 'message': 'Report exported successfully'})
        
    except Exception as e:
        error_msg = f"Export error: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500

@app.route('/api/upload', methods=['POST'])
def upload_documents():
    """Upload documents to the system"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        upload_dir = Path("data/uploaded_documents")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        uploaded_count = 0
        
        for file in files:
            if file.filename and file.filename.endswith('.txt'):
                filepath = upload_dir / file.filename
                file.save(str(filepath))
                uploaded_count += 1
                logger.info(f"Uploaded file: {file.filename}")
        
        # Try to reload documents in agents if they support it
        try:
            if research_agent and hasattr(research_agent, 'load_documents_from_directory'):
                research_agent.load_documents_from_directory(str(upload_dir))
            if super_agent and hasattr(super_agent, 'load_documents_from_directory'):
                super_agent.load_documents_from_directory(str(upload_dir))
        except Exception as e:
            logger.warning(f"Could not reload documents: {e}")
        
        return jsonify({
            'message': f'Successfully uploaded {uploaded_count} documents',
            'count': uploaded_count
        })
        
    except Exception as e:
        error_msg = f"Upload error: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    try:
        # Initialize agents on startup
        logger.info("Initializing research agents...")
        initialize_agents()
        
        # Verify agents are working
        if research_agent:
            try:
                test_report = research_agent.research("test")
                logger.info("Agent test successful")
            except Exception as e:
                logger.warning(f"Agent test failed: {e}")
        
        # Run the app
        port = int(os.environ.get('PORT', 5000))
        logger.info(f"Starting Flask app on port {port}")
        app.run(host='0.0.0.0', port=port, debug=True)
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)