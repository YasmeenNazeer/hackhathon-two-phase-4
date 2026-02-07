'use client';

import { useState, useEffect, useRef } from 'react';
import { Plus, CheckCircle, Circle, Trash2, Edit3, Save, X, MessageCircle, Send, User, Bot, Minus, Maximize2 } from 'lucide-react';
import { v4 as uuidv4 } from 'uuid';
import Link from 'next/link';

interface Task {
  id: string;
  title: string;
  description?: string;
  is_completed: boolean;
  category?: string;
  due_date?: string;
  created_at: string;
  updated_at: string;
}

interface Message {
  id: string;
  message: string;
  sender: 'user' | 'agent';
  createdAt: string;
  tool_calls?: {
    tool_name: string;
    arguments: Record<string, any>;
    output?: string;
  }[];
}

interface ChatResponse {
  ai_message: string;
  tool_calls: {
    tool_name: string;
    arguments: Record<string, any>;
    output?: string;
  }[];
}

export default function TasksPage() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [newTask, setNewTask] = useState({ title: '', description: '' });
  const [userId, setUserId] = useState<string>('');
  const [isEditing, setIsEditing] = useState<string | null>(null);
  const [editData, setEditData] = useState({ title: '', description: '' });
  const [loading, setLoading] = useState(true);

  // Chat state
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isChatOpen, setIsChatOpen] = useState(false);
  const messagesEndRef = useRef<null | HTMLDivElement>(null);

  // Initialize user ID from localStorage or generate new one
  useEffect(() => {
    let storedUserId = localStorage.getItem('userId');
    if (!storedUserId) {
      storedUserId = `user_${uuidv4()}`;
      localStorage.setItem('userId', storedUserId);
    }
    setUserId(storedUserId);
  }, []);

  const fetchTasks = async () => {
    if (!userId) return;

    try {
      setLoading(true);
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8002'}/api/tasks`, {
        headers: {
          'X-User-ID': userId,
        },
      });

      if (response.ok) {
        const data: Task[] = await response.json();
        setTasks(data);
      }
    } catch (error) {
      console.error('Error fetching tasks:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (userId) {
      fetchTasks();
    }
  }, [userId]);

  // Fetch chat history when chat is opened
  useEffect(() => {
    if (isChatOpen && userId) {
      fetchChatHistory();
    }
  }, [isChatOpen, userId]);

  const fetchChatHistory = async () => {
    if (!userId) return;

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8002'}/api/chat/${userId}/chat/history`, {
        headers: {
          'X-User-ID': userId,
        },
      });

      if (response.ok) {
        const history: any[] = await response.json();
        const formattedMessages: Message[] = history.map((msg, index) => ({
          id: `history-${index}`,
          message: msg.message,
          sender: msg.sender as 'user' | 'agent',
          createdAt: msg.created_at,
          tool_calls: msg.tool_calls || [],
        }));
        setMessages(formattedMessages);
      }
    } catch (error) {
      console.error('Error fetching chat history:', error);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!inputMessage.trim() || !userId || isLoading) return;
    if (inputMessage.length > 2000) {
      alert('Message is too long. Maximum length is 2000 characters.');
      return;
    }

    const userMessage: Message = {
      id: uuidv4(),
      message: inputMessage,
      sender: 'user',
      createdAt: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8002'}/api/chat/${userId}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-User-ID': userId,
        },
        body: JSON.stringify({ message: inputMessage }),
      });

      if (response.ok) {
        const data: ChatResponse = await response.json();
        const aiMessage: Message = {
          id: uuidv4(),
          message: data.ai_message,
          sender: 'agent',
          createdAt: new Date().toISOString(),
          tool_calls: data.tool_calls,
        };
        setMessages(prev => [...prev, aiMessage]);

        // Refresh tasks after AI interaction (in case tasks were modified)
        fetchTasks();
      } else {
        const errorData = await response.json();
        setMessages(prev => [
          ...prev,
          {
            id: uuidv4(),
            message: `Error: ${errorData.detail || 'Failed to get response'}`,
            sender: 'agent',
            createdAt: new Date().toISOString(),
          }
        ]);
      }
    } catch (error: any) {
      setMessages(prev => [
        ...prev,
        {
          id: uuidv4(),
          message: `Error: ${error.message || 'Failed to connect to the chat server'}`,
          sender: 'agent',
          createdAt: new Date().toISOString(),
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const createTask = async () => {
    if (!newTask.title.trim() || !userId) return;
    if (newTask.title.length > 200) {
      alert('Task title is too long. Maximum length is 200 characters.');
      return;
    }
    if (newTask.description && newTask.description.length > 1000) {
      alert('Task description is too long. Maximum length is 1000 characters.');
      return;
    }

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8002'}/api/tasks`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-User-ID': userId,
        },
        body: JSON.stringify({
          title: newTask.title,
          description: newTask.description,
        }),
      });

      if (response.ok) {
        const createdTask: Task = await response.json();
        setTasks([createdTask, ...tasks]);
        setNewTask({ title: '', description: '' });
      } else {
        const errorData = await response.json();
        alert(`Error creating task: ${errorData.detail || 'Unknown error'}`);
      }
    } catch (error: any) {
      console.error('Error creating task:', error);
      alert(`Error creating task: ${error.message || 'Network error'}`);
    }
  };

  const updateTask = async (taskId: string, updates: Partial<Task>) => {
    if (updates.title && updates.title.length > 200) {
      alert('Task title is too long. Maximum length is 200 characters.');
      return;
    }
    if (updates.description && updates.description.length > 1000) {
      alert('Task description is too long. Maximum length is 1000 characters.');
      return;
    }

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8002'}/api/tasks/${taskId}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'X-User-ID': userId,
        },
        body: JSON.stringify(updates),
      });

      if (response.ok) {
        const updatedTask: Task = await response.json();
        setTasks(tasks.map(task => task.id === taskId ? updatedTask : task));
        setIsEditing(null);
      } else {
        const errorData = await response.json();
        alert(`Error updating task: ${errorData.detail || 'Unknown error'}`);
      }
    } catch (error: any) {
      console.error('Error updating task:', error);
      alert(`Error updating task: ${error.message || 'Network error'}`);
    }
  };

  const toggleTaskComplete = async (taskId: string) => {
    const task = tasks.find(t => t.id === taskId);
    if (!task) return;

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8002'}/api/tasks/${taskId}/complete`, {
        method: 'PATCH',
        headers: {
          'X-User-ID': userId,
        },
      });

      if (response.ok) {
        const updatedTask: Task = await response.json();
        setTasks(tasks.map(task => task.id === taskId ? updatedTask : task));
      }
    } catch (error) {
      console.error('Error toggling task completion:', error);
    }
  };

  const deleteTask = async (taskId: string) => {
    if (!confirm('Are you sure you want to delete this task?')) return;

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8002'}/api/tasks/${taskId}`, {
        method: 'DELETE',
        headers: {
          'X-User-ID': userId,
        },
      });

      if (response.ok) {
        setTasks(tasks.filter(task => task.id !== taskId));
      }
    } catch (error) {
      console.error('Error deleting task:', error);
    }
  };

  const handleEdit = (task: Task) => {
    setIsEditing(task.id);
    setEditData({
      title: task.title,
      description: task.description || '',
    });
  };

  const saveEdit = (taskId: string) => {
    if (!editData.title.trim()) return;

    updateTask(taskId, {
      title: editData.title,
      description: editData.description,
    });
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-950 text-white flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-mulberry mb-4"></div>
          <p>Loading tasks...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-950 text-white flex flex-col">
      {/* Header */}
      <header className="border-b border-gray-800 py-4 px-6 flex justify-between items-center">
        <div className="flex items-center gap-2">
          <div className="p-2 bg-mulberry rounded-xl shadow-lg shadow-mulberry/20">
            <Plus className="h-5 w-5 text-white" />
          </div>
          <span className="text-xl font-bold tracking-tighter uppercase italic">Elevate</span>
        </div>

        <nav className="flex items-center gap-6">
          <Link href="/" className="text-sm font-bold uppercase tracking-widest text-gray-400 hover:text-mulberry transition-colors">
            Home
          </Link>
          <Link href="/tasks" className="text-sm font-bold uppercase tracking-widest text-mulberry">
            Tasks
          </Link>
        </nav>
      </header>

      {/* Tasks Container */}
      <div className="flex-1 flex flex-col max-w-4xl w-full mx-auto p-4">
        <div className="flex items-center gap-3 mb-6">
          <Plus className="h-6 w-6 text-mulberry" />
          <h1 className="text-2xl font-bold">Your Tasks</h1>
        </div>

        {/* Add Task Form */}
        <div className="bg-gray-800/50 backdrop-blur-lg p-4 rounded-2xl border border-gray-700 mb-6">
          <h2 className="text-lg font-semibold mb-3">Add New Task</h2>
          <div className="space-y-3">
            <input
              type="text"
              value={newTask.title}
              onChange={(e) => setNewTask({...newTask, title: e.target.value})}
              placeholder="Task title..."
              className="w-full bg-gray-900/50 border border-gray-700 rounded-xl px-4 py-2 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-mulberry focus:border-transparent"
              onKeyDown={(e) => e.key === 'Enter' && createTask()}
            />
            <textarea
              value={newTask.description}
              onChange={(e) => setNewTask({...newTask, description: e.target.value})}
              placeholder="Task description (optional)..."
              className="w-full bg-gray-900/50 border border-gray-700 rounded-xl px-4 py-2 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-mulberry focus:border-transparent resize-none"
              rows={2}
            />
            <button
              onClick={createTask}
              disabled={!newTask.title.trim()}
              className="bg-mulberry hover:bg-mulberry-light disabled:opacity-50 disabled:cursor-not-allowed px-4 py-2 rounded-lg transition-colors"
            >
              Add Task
            </button>
          </div>
        </div>

        {/* Tasks List */}
        <div>
          <h2 className="text-lg font-semibold mb-3">Your Tasks ({tasks.filter(t => !t.is_completed).length} pending)</h2>

          {tasks.length === 0 ? (
            <div className="bg-gray-800/50 backdrop-blur-lg p-8 rounded-2xl border border-gray-700 text-center">
              <p className="text-gray-400">No tasks yet. Add a new task above or start chatting with your AI assistant!</p>
            </div>
          ) : (
            <div className="space-y-3">
              {tasks.map((task) => (
                <div
                  key={task.id}
                  className={`bg-gray-800/50 backdrop-blur-lg p-4 rounded-2xl border border-gray-700 ${
                    task.is_completed ? 'opacity-70' : ''
                  }`}
                >
                  {isEditing === task.id ? (
                    <div className="space-y-3">
                      <input
                        type="text"
                        value={editData.title}
                        onChange={(e) => setEditData({...editData, title: e.target.value})}
                        className="w-full bg-gray-900/50 border border-gray-700 rounded-xl px-3 py-1 text-white mb-2"
                      />
                      <textarea
                        value={editData.description}
                        onChange={(e) => setEditData({...editData, description: e.target.value})}
                        className="w-full bg-gray-900/50 border border-gray-700 rounded-xl px-3 py-1 text-white resize-none"
                        rows={2}
                      />
                      <div className="flex gap-2">
                        <button
                          onClick={() => saveEdit(task.id)}
                          className="flex items-center gap-1 bg-green-600 hover:bg-green-700 px-3 py-1 rounded-lg text-sm"
                        >
                          <Save className="h-4 w-4" /> Save
                        </button>
                        <button
                          onClick={() => setIsEditing(null)}
                          className="flex items-center gap-1 bg-gray-600 hover:bg-gray-700 px-3 py-1 rounded-lg text-sm"
                        >
                          <X className="h-4 w-4" /> Cancel
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-start gap-3">
                      <button
                        onClick={() => toggleTaskComplete(task.id)}
                        className="mt-0.5"
                      >
                        {task.is_completed ? (
                          <CheckCircle className="h-5 w-5 text-green-500" />
                        ) : (
                          <Circle className="h-5 w-5 text-gray-500 hover:text-mulberry" />
                        )}
                      </button>

                      <div className="flex-1">
                        <h3 className={`font-medium ${task.is_completed ? 'line-through' : ''}`}>
                          {task.title}
                        </h3>
                        {task.description && (
                          <p className="text-gray-400 text-sm mt-1">{task.description}</p>
                        )}
                        <div className="flex items-center gap-2 mt-2 text-xs text-gray-500">
                          {task.category && <span className="px-2 py-1 bg-gray-700 rounded-full">{task.category}</span>}
                          {task.due_date && (
                            <span>Due: {new Date(task.due_date).toLocaleDateString()}</span>
                          )}
                          <span>Created: {new Date(task.created_at).toLocaleDateString()}</span>
                        </div>
                      </div>

                      <div className="flex gap-1">
                        <button
                          onClick={() => handleEdit(task)}
                          className="p-2 text-gray-400 hover:text-white rounded-lg hover:bg-gray-700"
                        >
                          <Edit3 className="h-4 w-4" />
                        </button>
                        <button
                          onClick={() => deleteTask(task.id)}
                          className="p-2 text-gray-400 hover:text-red-500 rounded-lg hover:bg-gray-700"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    </div>
                  )}
                </div>
            ))}
          </div>
        )}
      </div>
    </div>

    {/* Floating Chat Button */}
      <button
        onClick={() => setIsChatOpen(!isChatOpen)}
        className="fixed bottom-6 right-6 bg-mulberry hover:bg-mulberry-light p-4 rounded-full shadow-lg shadow-mulberry/50 transition-all duration-300 z-50 border-2 border-white/20"
      >
        {isChatOpen ? <Minus className="h-6 w-6 text-white" /> : <MessageCircle className="h-6 w-6 text-white" />}
      </button>

      {/* Chat Panel */}
      {isChatOpen && (
        <div className="fixed bottom-24 right-6 w-96 h-96 bg-gray-800/90 backdrop-blur-lg rounded-2xl border border-gray-700 shadow-xl z-50 flex flex-col">
          {/* Chat Header */}
          <div className="p-4 border-b border-gray-700 flex justify-between items-center">
            <div className="flex items-center gap-2">
              <Bot className="h-5 w-5 text-mulberry" />
              <h3 className="font-bold">AI Assistant</h3>
            </div>
            <button
              onClick={() => setIsChatOpen(false)}
              className="text-gray-400 hover:text-white"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          {/* Messages Container */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center text-gray-400">
                <MessageCircle className="h-10 w-10 mb-2" />
                <p>Start a conversation with your AI assistant</p>
                <p className="text-sm mt-1">Ask to create, update, or manage tasks</p>
              </div>
            ) : (
              messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[80%] rounded-2xl p-3 ${
                      msg.sender === 'user'
                        ? 'bg-mulberry rounded-br-none'
                        : 'bg-gray-700 rounded-bl-none'
                    }`}
                  >
                    <div className="flex items-start gap-2">
                      <div className="mt-0.5">
                        {msg.sender === 'user' ? (
                          <User className="h-4 w-4 text-white" />
                        ) : (
                          <Bot className="h-4 w-4 text-mulberry" />
                        )}
                      </div>
                      <div className="flex-1">
                        <p className="whitespace-pre-wrap text-sm">{msg.message}</p>

                        {/* Tool Calls Display */}
                        {msg.tool_calls && msg.tool_calls.length > 0 && (
                          <div className="mt-2 pt-2 border-t border-gray-600">
                            <div className="text-xs text-gray-400 mb-1">Tool Executions:</div>
                            {msg.tool_calls.map((toolCall, idx) => (
                              <div key={idx} className="text-xs bg-gray-800 p-2 rounded mt-1">
                                <div className="font-mono text-cocoa">
                                  {toolCall.tool_name}({JSON.stringify(toolCall.arguments)})
                                </div>
                                {toolCall.output && (
                                  <div className="mt-1 text-gray-300">
                                    <span className="text-xs">Result:</span> {toolCall.output}
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              ))
            )}
            {isLoading && (
              <div className="flex justify-start">
                <div className="max-w-[80%] rounded-2xl p-3 bg-gray-700 rounded-bl-none">
                  <div className="flex items-center gap-2">
                    <Bot className="h-4 w-4 text-mulberry" />
                    <div className="flex space-x-1">
                      <div className="h-2 w-2 bg-mulberry rounded-full animate-bounce"></div>
                      <div className="h-2 w-2 bg-mulberry rounded-full animate-bounce delay-100"></div>
                      <div className="h-2 w-2 bg-mulberry rounded-full animate-bounce delay-200"></div>
                    </div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="p-3 border-t border-gray-700">
            <div className="flex gap-2">
              <textarea
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyDown={handleKeyPress}
                placeholder="Ask me to manage tasks..."
                className="flex-1 bg-gray-900/50 border border-gray-700 rounded-xl px-3 py-2 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-mulberry focus:border-transparent resize-none text-sm"
                rows={2}
                disabled={isLoading}
              />
              <button
                onClick={sendMessage}
                disabled={isLoading || !inputMessage.trim()}
                className="bg-mulberry hover:bg-mulberry-light disabled:opacity-50 disabled:cursor-not-allowed p-3 rounded-xl transition-colors self-end"
              >
                <Send className="h-4 w-4 text-white" />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}