import React, { useState, useRef, useEffect } from 'react';
import { Send, Terminal, Cpu, Database, Activity, Code } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import axios from 'axios';
import './App.css';

interface Message {
  role: 'user' | 'ai';
  content: string;
  model?: string;
}

// Updated interface to include tokens
interface RoutingLog {
  query: string;
  classification: string;
  model_used: string;
  tokens_input: number;
  tokens_output: number;
  latency_ms: number;
  timestamp: string;
}

interface ChatResponse {
  answer: string;
  confidence_flag: boolean;
  flag_reason: string;
  model_used: string;
  tokens_input: number;
  tokens_output: number;
  latency_ms: number;
}

const App: React.FC = () => {
  const [query, setQuery] = useState<string>('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [logs, setLogs] = useState<RoutingLog[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || loading) return;

    const currentQuery = query;
    setMessages(prev => [...prev, { role: 'user', content: currentQuery }]);
    setLoading(true);
    setQuery('');

    try {
      const { data } = await axios.post<ChatResponse>('http://localhost:8000/api/chat', { 
        query: currentQuery 
      });
      
      setMessages(prev => [...prev, { 
        role: 'ai', 
        content: data.answer, 
        model: data.model_used 
      }]);
      
      setLogs(prev => [{
        query: currentQuery,
        classification: data.flag_reason,
        model_used: data.model_used,
        tokens_input: data.tokens_input,
        tokens_output: data.tokens_output,
        latency_ms: data.latency_ms,
        timestamp: new Date().toLocaleTimeString()
      }, ...prev]);

    } catch (err) {
      setMessages(prev => [...prev, { 
        role: 'ai', 
        content: "❌ **Error:** Connection to FastAPI failed." 
      }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>ClearPath-RAG-ChatBot</h1>
        <p>What's in your mind?</p>
      </header>

      <main className="main-layout">
        <section className="chat-section">
          <div className="messages-container">
            {messages.length === 0 && (
              <div className="empty-state">
                <Database size={48} className="pulse-icon" />
                <p>Systems Online. Ready for your query.</p>
              </div>
            )}
            
            {messages.map((m, i) => (
              <div key={i} className={`message ${m.role === 'user' ? 'user-message' : 'ai-message'}`}>
                <ReactMarkdown>{m.content}</ReactMarkdown>
                {m.model && (
                  <div className="model-tag">
                    <Activity size={10} /> {m.model}
                  </div>
                )}
              </div>
            ))}
            {loading && <div className="ai-message loading-text">Thinking...</div>}
            <div ref={scrollRef} />
          </div>

          <form className="input-area" onSubmit={handleSend}>
            <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Type here..." autoFocus />
            <button type="submit" disabled={loading}><Send size={18} /></button>
          </form>
        </section>

        {/* LOGS SIDEBAR WITH KEY-VALUE FORMAT */}
        <aside className="logs-sidebar">
          <div className="logs-header">
            <Terminal size={18} color="#a78bfa" />
            <span>LIVE TRACES</span>
          </div>
          
          <div className="logs-list">
            {logs.map((log, i) => (
              <div key={i} className="json-log">
                <div className="log-timestamp">// {log.timestamp}</div>
                <div className="json-block">
                  <span className="json-key">"query"</span>: <span className="json-string">"{log.query}"</span>,
                  <br />
                  <span className="json-key">"classification"</span>: <span className="json-string">"{log.classification}"</span>,
                  <br />
                  <span className="json-key">"model_used"</span>: <span className="json-string">"{log.model_used}"</span>,
                  <br />
                  <span className="json-key">"tokens_in"</span>: <span className="json-num">{log.tokens_input}</span>,
                  <br />
                  <span className="json-key">"tokens_out"</span>: <span className="json-num">{log.tokens_output}</span>,
                  <br />
                  <span className="json-key">"latency"</span>: <span className="json-num">{log.latency_ms}</span>
                </div>
              </div>
            ))}
          </div>
        </aside>
      </main>
    </div>
  );
};

export default App;