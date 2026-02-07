"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import api from "@/lib/api";
import { useSession, signOut } from "@/lib/auth-client";
import { useRouter } from "next/navigation";
import { useEffect, useState, useMemo } from "react";
import { LogOut, Plus, Trash2, CheckCircle, Circle, Loader2, AlertCircle, Pencil, X, Save, Tag, Filter, Calendar, Bell, Clock, Search, SortAsc, SortDesc, Sparkles, Menu, User, Sun, Moon, BarChart3, TrendingUp, CheckCircle2 } from "lucide-react";
import Link from "next/link";

interface Task {
  id: string;
  title: string;
  description?: string;
  is_completed: boolean;
  category: string;
  due_date?: string;
  created_at: string;
}

const CATEGORIES = ["Personal", "Work", "Urgent", "Shopping", "Health"];

export default function DashboardPage() {
  const { data: session, isPending: isSessionPending } = useSession();
  const router = useRouter();
  const queryClient = useQueryClient();
  const [newTitle, setNewTitle] = useState("");
  const [newDescription, setNewDescription] = useState("");
  const [newCategory, setNewCategory] = useState("Personal");
  const [newDueDate, setNewDueDate] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const [editDescription, setEditDescription] = useState("");
  const [editCategory, setEditCategory] = useState("Personal");
  const [editDueDate, setEditDueDate] = useState("");
  const [filterCategory, setFilterCategory] = useState("All");
  const [searchQuery, setSearchQuery] = useState("");
  const [sortBy, setSortBy] = useState<"created_at" | "due_date" | "title">("created_at");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(true);

  useEffect(() => {
    if (!isSessionPending && !session) {
      router.push("/login");
    }
    // Initialize dark mode from system or local storage
    const savedTheme = localStorage.getItem("theme");
    if (savedTheme === "light") {
        setIsDarkMode(false);
        document.documentElement.classList.remove("dark");
    } else {
        setIsDarkMode(true);
        document.documentElement.classList.add("dark");
    }
  }, [session, isSessionPending, router]);

  const toggleTheme = () => {
    const newMode = !isDarkMode;
    setIsDarkMode(newMode);
    if (newMode) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  };

  const { data: tasks, isLoading: isTasksLoading, error: fetchError } = useQuery({
    queryKey: ["tasks"],
    queryFn: async () => {
      const res = await api.get<Task[]>("/tasks");
      return res.data;
    },
    enabled: !!session,
    retry: false
  });

  // Analytics Calculations
  const analytics = useMemo(() => {
    if (!tasks) return null;
    const total = tasks.length;
    const completed = tasks.filter(t => t.is_completed).length;
    const pending = total - completed;
    const completionRate = total > 0 ? Math.round((completed / total) * 100) : 0;

    // Category distribution
    const categoryStats = CATEGORIES.map(cat => ({
        name: cat,
        count: tasks.filter(t => t.category === cat).length,
        completed: tasks.filter(t => t.category === cat && t.is_completed).length
    }));

    return { total, completed, pending, completionRate, categoryStats };
  }, [tasks]);

  const getFilteredAndSortedTasks = () => {
    if (!tasks) return [];

    let filtered = tasks.filter(task => {
      const matchesCategory = filterCategory === "All" || task.category === filterCategory;
      const matchesSearch = task.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           (task.description?.toLowerCase().includes(searchQuery.toLowerCase()));
      return matchesCategory && matchesSearch;
    });

    return filtered.sort((a, b) => {
      let comparison = 0;
      if (sortBy === "title") {
        comparison = a.title.localeCompare(b.title);
      } else {
        const dateA = a[sortBy] ? new Date(a[sortBy]!).getTime() : (sortOrder === "asc" ? Infinity : -Infinity);
        const dateB = b[sortBy] ? new Date(b[sortBy]!).getTime() : (sortOrder === "asc" ? Infinity : -Infinity);
        comparison = dateA - dateB;
      }
      return sortOrder === "asc" ? comparison : -comparison;
    });
  };

  const filteredTasks = getFilteredAndSortedTasks();

  const createTask = useMutation({
    mutationFn: async (newTask: { title: string; description?: string, category: string, due_date?: string }) => {
      return api.post("/tasks", newTask);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["tasks"] });
      setNewTitle("");
      setNewDescription("");
      setNewCategory("Personal");
      setNewDueDate("");
    },
    onError: (error: any) => {
      const detail = error.response?.data?.detail || error.message || "Unknown Error";
      alert("Error: " + detail);
    }
  });

  const updateTask = useMutation({
    mutationFn: async ({ id, title, description, category, due_date }: { id: string; title: string, description?: string, category: string, due_date?: string }) => {
      return api.put(`/tasks/${id}`, { title, description, category, due_date: due_date || null });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["tasks"] });
      setEditingId(null);
    },
    onError: (error: any) => {
      const detail = error.response?.data?.detail || error.message || "Unknown Error";
      alert("Update Failed: " + detail);
    }
  });

  const toggleComplete = useMutation({
    mutationFn: async (id: string) => api.patch(`/tasks/${id}/complete`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["tasks"] }),
  });

  const deleteTask = useMutation({
    mutationFn: async (id: string) => api.delete(`/tasks/${id}`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["tasks"] }),
  });

  const startEditing = (task: Task) => {
    setEditingId(task.id);
    setEditTitle(task.title);
    setEditDescription(task.description || "");
    setEditCategory(task.category || "Personal");
    setEditDueDate(task.due_date ? new Date(task.due_date).toISOString().slice(0, 16) : "");
  };

  const getCategoryColor = (cat: string) => {
    switch (cat) {
      case "Urgent": return "bg-red-500/10 text-red-500 dark:text-red-400 border-red-500/20";
      case "Work": return "bg-blue-500/10 text-blue-500 dark:text-blue-400 border-blue-500/20";
      case "Shopping": return "bg-orange-500/10 text-orange-500 dark:text-orange-400 border-orange-500/20";
      case "Health": return "bg-green-500/10 text-green-500 dark:text-green-400 border-green-500/20";
      default: return "bg-gray-100 dark:bg-white/5 text-gray-500 dark:text-gray-400 border-gray-200 dark:border-white/10";
    }
  };

  const isOverdue = (dateStr: string) => {
    return new Date(dateStr) < new Date();
  };

  const formatDueDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  };

  if (isSessionPending) return (
    <div className="flex min-h-screen items-center justify-center bg-background">
      <Loader2 className="h-10 w-10 animate-spin text-mulberry" />
    </div>
  );

  if (!session) return null;

  return (
    <div className="min-h-screen bg-background text-foreground transition-colors duration-300">
      {/* Background Orbs */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none opacity-10 dark:opacity-20 transition-opacity duration-500">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-mulberry rounded-full blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-cocoa rounded-full blur-[120px]" />
      </div>

      <nav className="border-b border-black/5 dark:border-white/5 sticky top-0 z-50 backdrop-blur-xl bg-background/80 transition-colors duration-300">
        <div className="mx-auto max-w-5xl px-6 h-20 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-mulberry rounded-xl shadow-lg shadow-mulberry/20">
              <Sparkles className="h-6 w-6 text-white" />
            </div>
            <h1 className="text-2xl font-bold tracking-tight text-gray-900 dark:text-white transition-colors">
              Elevate
            </h1>
          </div>

          {/* Desktop Nav */}
          <div className="hidden md:flex items-center gap-4 lg:gap-6">
            <button
                onClick={toggleTheme}
                className="p-3 bg-black/5 dark:bg-white/5 text-gray-500 dark:text-gray-400 rounded-2xl border border-black/5 dark:border-white/5 hover:bg-mulberry/10 hover:text-mulberry transition-all duration-300"
                title={`Switch to ${isDarkMode ? 'Light' : 'Dark'} Mode`}
            >
                {isDarkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
            </button>

            <Link href="/profile" className="flex flex-col items-end hover:opacity-80 transition-opacity">
              <span className="text-sm font-semibold text-gray-900 dark:text-white">{session.user.name || "User"}</span>
              <span className="text-xs text-gray-400 dark:text-gray-500">{session.user.email}</span>
            </Link>
            <button onClick={() => { signOut(); router.push("/login"); }}
                    className="p-3 bg-black/5 dark:bg-white/5 hover:bg-red-500/10 text-gray-500 dark:text-gray-400 hover:text-red-500 rounded-2xl transition-all duration-300 border border-black/5 dark:border-white/5 hover:border-red-500/20 group">
              <LogOut className="h-5 w-5 group-hover:scale-110 transition-transform" />
            </button>
          </div>

          {/* Mobile Toggle Icons */}
          <div className="flex items-center gap-2 md:hidden">
              <button
                onClick={toggleTheme}
                className="p-3 bg-black/5 dark:bg-white/5 text-gray-500 dark:text-gray-400 rounded-2xl border border-black/5 dark:border-white/5 transition-all"
              >
                {isDarkMode ? <Sun className="h-6 w-6" /> : <Moon className="h-6 w-6" />}
              </button>
              <button
                onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                className="p-3 bg-black/5 dark:bg-white/5 text-gray-500 dark:text-gray-400 rounded-2xl border border-black/5 dark:border-white/5"
              >
                {isMobileMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
              </button>
          </div>
        </div>

        {/* Mobile Navigation Menu */}
        {isMobileMenuOpen && (
          <div className="md:hidden absolute top-20 left-0 right-0 bg-background/95 backdrop-blur-2xl border-b border-black/10 dark:border-white/10 p-6 z-40 animate-in slide-in-from-top duration-300">
            <div className="space-y-6">
              <Link
                href="/profile"
                onClick={() => setIsMobileMenuOpen(false)}
                className="flex items-center gap-4 p-4 rounded-3xl bg-black/5 dark:bg-white/5 border border-black/5 dark:border-white/10"
              >
                <div className="p-3 bg-mulberry/20 rounded-2xl text-mulberry-light">
                  <User className="h-6 w-6" />
                </div>
                <div>
                  <p className="text-gray-900 dark:text-white font-bold">{session.user.name || "User"}</p>
                  <p className="text-gray-500 text-xs truncate max-w-[200px]">{session.user.email}</p>
                </div>
              </Link>

              <div className="grid grid-cols-2 gap-4">
                <button
                  onClick={() => setIsMobileMenuOpen(false)}
                  className="flex items-center justify-center gap-2 p-4 rounded-3xl bg-black/5 dark:bg-white/5 text-gray-500 dark:text-gray-400 border border-black/5 dark:border-white/5 font-bold text-sm"
                >
                  <Bell className="h-5 w-5" /> Alerts
                </button>
                <button
                  onClick={() => { signOut(); router.push("/login"); }}
                  className="flex items-center justify-center gap-2 p-4 rounded-3xl bg-red-500/10 text-red-500 dark:text-red-400 border border-red-500/20 font-bold text-sm"
                >
                  <LogOut className="h-5 w-5" /> Logout
                </button>
              </div>
            </div>
          </div>
        )}
      </nav>

      <main className="mx-auto max-w-5xl px-6 py-12 relative z-10">
        {/* Header Stats / Intro */}
        <div className="mb-12 flex flex-col md:flex-row md:items-end justify-between gap-6">
          <div>
            <h2 className="text-3xl md:text-4xl font-black tracking-tight mb-2 text-gray-900 dark:text-white transition-colors italic uppercase">Nexus Dashboard</h2>
            <p className="text-gray-500 font-medium">Manage your tasks and stay focused.</p>
          </div>
          <div className="flex items-center gap-4">
               <div className="p-6 bg-white dark:bg-white/5 backdrop-blur-md rounded-3xl border border-black/5 dark:border-white/10 shadow-xl flex items-center gap-4 group hover:border-mulberry/30 transition-all">
                  <div className="p-3 bg-mulberry/10 rounded-2xl text-mulberry transition-transform group-hover:scale-110">
                      <TrendingUp className="h-6 w-6" />
                  </div>
                  <div>
                      <p className="text-[10px] font-black text-gray-400 uppercase tracking-widest mb-1">Efficiency</p>
                      <p className="text-xl font-black text-gray-900 dark:text-white">{analytics?.completionRate}%</p>
                  </div>
               </div>
          </div>
        </div>

        {/* Analytics Section */}
        {analytics && (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
                <div className="p-6 bg-white dark:bg-white/5 backdrop-blur-md rounded-3xl border border-black/5 dark:border-white/10 shadow-md">
                   <div className="flex justify-between items-start mb-4">
                      <div className="p-2 bg-blue-500/10 rounded-xl text-blue-500">
                        <BarChart3 className="h-5 w-5" />
                      </div>
                      <span className="text-[10px] font-black text-gray-300 dark:text-gray-600 uppercase">Total</span>
                   </div>
                   <p className="text-3xl font-black text-gray-900 dark:text-white">{analytics.total}</p>
                   <p className="text-xs font-bold text-gray-400 mt-1 uppercase tracking-tighter">Created Tasks</p>
                </div>

                <div className="p-6 bg-white dark:bg-white/5 backdrop-blur-md rounded-3xl border border-black/5 dark:border-white/10 shadow-md">
                   <div className="flex justify-between items-start mb-4">
                      <div className="p-2 bg-green-500/10 rounded-xl text-green-500">
                        <CheckCircle2 className="h-5 w-5" />
                      </div>
                      <span className="text-[10px] font-black text-gray-300 dark:text-gray-600 uppercase">Success</span>
                   </div>
                   <p className="text-3xl font-black text-gray-900 dark:text-white">{analytics.completed}</p>
                   <p className="text-xs font-bold text-gray-400 mt-1 uppercase tracking-tighter">Done Tasks</p>
                </div>

                <div className="p-6 bg-white dark:bg-white/5 backdrop-blur-md rounded-3xl border border-black/5 dark:border-white/10 shadow-md">
                   <div className="flex justify-between items-start mb-4">
                      <div className="p-2 bg-orange-500/10 rounded-xl text-orange-500">
                        <Clock className="h-5 w-5" />
                      </div>
                      <span className="text-[10px] font-black text-gray-300 dark:text-gray-600 uppercase">Pending</span>
                   </div>
                   <p className="text-3xl font-black text-gray-900 dark:text-white">{analytics.pending}</p>
                   <p className="text-xs font-bold text-gray-400 mt-1 uppercase tracking-tighter">Tasks to Do</p>
                </div>

                <div className="p-6 bg-white dark:bg-white/5 backdrop-blur-md rounded-3xl border border-black/5 dark:border-white/10 shadow-md overflow-hidden relative group">
                   <div className="absolute top-0 right-0 p-2 bg-mulberry/10 rounded-bl-2xl opacity-0 group-hover:opacity-100 transition-opacity">
                      <Sparkles className="h-4 w-4 text-mulberry" />
                   </div>
                   <div className="flex justify-between items-start mb-4">
                      <div className="p-2 bg-mulberry/10 rounded-xl text-mulberry">
                        <TrendingUp className="h-5 w-5" />
                      </div>
                      <span className="text-[10px] font-black text-gray-300 dark:text-gray-600 uppercase">Velocity</span>
                   </div>
                   <div className="w-full bg-gray-100 dark:bg-white/5 h-2 rounded-full mt-4 mb-2 overflow-hidden">
                      <div className="h-full bg-mulberry transition-all duration-1000" style={{ width: `${analytics.completionRate}%` }}></div>
                   </div>
                   <p className="text-xl font-black text-gray-900 dark:text-white">{analytics.completionRate}% <span className="text-[10px] text-gray-400 uppercase font-bold tracking-widest ml-1">Rate</span></p>
                </div>
            </div>
        )}

        {/* Create Task Form */}
        <form onSubmit={(e) => { e.preventDefault(); if (newTitle.trim()) createTask.mutate({ title: newTitle, description: newDescription, category: newCategory, due_date: newDueDate || undefined }); }}
              className="bg-gray-50 dark:bg-white/5 backdrop-blur-md p-6 md:p-8 rounded-[2rem] md:rounded-3xl border border-black/5 dark:border-white/10 mb-12 md:mb-16 shadow-2xl transition-colors duration-300">
          <div className="space-y-6">
            <div className="space-y-2">
               <label className="text-xs font-bold text-gray-400 dark:text-gray-500 uppercase tracking-[0.2em] ml-1">Task Title</label>
               <input
                className="w-full bg-white dark:bg-white/5 p-4 border border-gray-200 dark:border-white/10 rounded-2xl focus:ring-2 focus:ring-mulberry outline-none transition-all text-lg placeholder:text-gray-400 dark:placeholder:text-gray-600 dark:text-white text-gray-900"
                placeholder="What is your next task?"
                value={newTitle}
                onChange={(e) => setNewTitle(e.target.value)}
                required
              />
            </div>

            <div className="space-y-2">
              <label className="text-xs font-bold text-gray-400 dark:text-gray-500 uppercase tracking-[0.2em] ml-1">Task Details</label>
              <textarea
                className="w-full bg-white dark:bg-white/5 p-4 border border-gray-200 dark:border-white/10 rounded-2xl focus:ring-2 focus:ring-mulberry outline-none transition-all placeholder:text-gray-400 dark:placeholder:text-gray-600 dark:text-white text-gray-900 resize-none"
                placeholder="Add more details here..."
                value={newDescription}
                onChange={(e) => setNewDescription(e.target.value)}
                rows={3}
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
               {/* Category Picker */}
               <div className="space-y-3">
                 <label className="text-xs font-bold text-gray-400 dark:text-gray-500 uppercase tracking-[0.2em] ml-1 flex items-center gap-2"><Tag className="h-3 w-3" /> Category</label>
                 <div className="flex flex-wrap items-center gap-2">
                  {CATEGORIES.map(cat => (
                    <button
                      key={cat}
                      type="button"
                      onClick={() => setNewCategory(cat)}
                      className={`px-4 md:px-5 py-2 rounded-2xl text-xs font-bold border transition-all duration-300 ${
                        newCategory === cat ? "bg-mulberry text-white border-mulberry shadow-lg shadow-mulberry/20 scale-105" : "bg-white dark:bg-white/5 text-gray-500 border-gray-200 dark:border-white/10 hover:border-mulberry/40 shadow-sm"
                      }`}
                    >
                      {cat}
                    </button>
                  ))}
                 </div>
               </div>

               {/* Due Date Picker */}
               <div className="space-y-3">
                 <label className="text-xs font-bold text-gray-400 dark:text-gray-500 uppercase tracking-[0.2em] ml-1 flex items-center gap-2"><Calendar className="h-3 w-3" /> Due Date</label>
                 <input
                   type="datetime-local"
                   className="w-full bg-white dark:bg-white/5 p-4 border border-gray-200 dark:border-white/10 rounded-2xl text-sm font-medium text-gray-700 dark:text-gray-400 focus:ring-2 focus:ring-mulberry outline-none transition-all"
                   value={newDueDate}
                   onChange={(e) => setNewDueDate(e.target.value)}
                 />
               </div>
            </div>

            <button
              disabled={createTask.isPending}
              className="w-full bg-mulberry text-white py-5 rounded-2xl font-bold hover:bg-mulberry-light disabled:opacity-50 flex justify-center items-center gap-3 shadow-xl shadow-mulberry/20 transition-all duration-300 hover:scale-[1.01] active:scale-[0.99] text-lg uppercase tracking-widest"
            >
              {createTask.isPending ? <Loader2 className="h-6 w-6 animate-spin" /> : <Plus className="h-6 w-6" />}
              ADD NEW TASK
            </button>
          </div>
        </form>

        {/* Search and Filters Section */}
        <div className="bg-gray-50 dark:bg-white/5 backdrop-blur-sm p-6 rounded-[2rem] md:rounded-3xl border border-black/5 dark:border-white/10 mb-12 flex flex-col lg:flex-row gap-6 items-center shadow-lg transition-colors">
            {/* Search Bar */}
            <div className="relative flex-1 group w-full">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400 dark:text-gray-500 group-focus-within:text-mulberry transition-colors" />
              <input
                type="text"
                placeholder="Search for a task..."
                className="w-full pl-12 pr-6 py-4 bg-white/50 dark:bg-background/50 border border-gray-200 dark:border-white/5 rounded-2xl text-sm focus:bg-white dark:focus:bg-background/80 focus:ring-2 focus:ring-mulberry outline-none transition-all placeholder:text-gray-400 dark:placeholder:text-gray-600 dark:text-white text-gray-900"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>

            <div className="flex flex-col sm:flex-row items-center gap-3 w-full lg:w-auto">
              {/* Category Filter */}
              <div className="flex items-center gap-3 bg-white/50 dark:bg-background/50 px-4 py-2 rounded-2xl border border-gray-100 dark:border-white/5 w-full sm:w-auto flex-1 md:flex-none">
                <Filter className="h-4 w-4 text-gray-400 dark:text-gray-500" />
                <select
                  className="text-xs font-bold text-gray-500 dark:text-gray-400 outline-none bg-transparent py-2 w-full cursor-pointer"
                  value={filterCategory}
                  onChange={(e) => setFilterCategory(e.target.value)}
                >
                  <option value="All" className="bg-white dark:bg-background text-gray-900 dark:text-white">All Categories</option>
                  {CATEGORIES.map(cat => <option key={cat} value={cat} className="bg-white dark:bg-background text-gray-900 dark:text-white">{cat}</option>)}
                </select>
              </div>

              {/* Sort Dropdown */}
              <div className="flex items-center gap-3 bg-white/50 dark:bg-background/50 px-4 py-2 rounded-2xl border border-gray-100 dark:border-white/5 w-full sm:w-auto flex-1 md:flex-none">
                <SortAsc className="h-4 w-4 text-gray-400 dark:text-gray-500" />
                <select
                  className="text-xs font-bold text-gray-500 dark:text-gray-400 outline-none bg-transparent py-2 w-full cursor-pointer"
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value as any)}
                >
                  <option value="created_at" className="bg-white dark:bg-background text-gray-900 dark:text-white">Date Created</option>
                  <option value="due_date" className="bg-white dark:bg-background text-gray-900 dark:text-white">Due Date</option>
                  <option value="title" className="bg-white dark:bg-background text-gray-900 dark:text-white">Title</option>
                </select>
                <button
                  onClick={() => setSortOrder(prev => prev === "asc" ? "desc" : "asc")}
                  className="p-2 hover:bg-black/5 dark:hover:bg-white/5 rounded-xl transition-all duration-300"
                >
                  {sortOrder === "asc" ? <SortAsc className="h-4 w-4 text-mulberry" /> : <SortDesc className="h-4 w-4 text-mulberry" />}
                </button>
              </div>
            </div>
        </div>

        {fetchError && (
          <div className="bg-red-500/10 border border-red-500/20 p-6 rounded-3xl flex items-center gap-4 text-red-500 dark:text-red-400 mb-12 font-medium">
            <AlertCircle className="h-6 w-6" />
            <p>Could not connect to the server. Checking connection...</p>
          </div>
        )}

        {/* Task List Section */}
        <div className="space-y-6">
          <div className="flex justify-between items-center px-4 mb-4">
            <h2 className="text-sm font-bold text-gray-400 dark:text-gray-500 uppercase tracking-[0.3em]">Your Tasks</h2>
            <span className="text-[10px] text-mulberry font-bold bg-mulberry/10 px-3 py-1 rounded-full border border-mulberry/20 uppercase tracking-widest transition-all">{filteredTasks?.length || 0} Tasks Found</span>
          </div>

          {isTasksLoading ? (
            <div className="flex flex-col items-center justify-center py-32 gap-6 bg-gray-50 dark:bg-white/5 rounded-[2rem] border border-black/5 dark:border-white/5">
              <Loader2 className="h-12 w-12 animate-spin text-mulberry/40" />
              <p className="text-xs text-gray-400 dark:text-gray-500 font-bold tracking-[0.4em] uppercase">Loading Tasks</p>
            </div>
          ) : filteredTasks?.length === 0 ? (
            <div className="text-center py-32 bg-gray-50 dark:bg-white/5 rounded-[2rem] border-2 border-dashed border-gray-100 dark:border-white/5 text-gray-400 dark:text-gray-600">
              {searchQuery ? (
                <div className="space-y-4">
                  <Search className="h-12 w-12 mx-auto opacity-20" />
                  <p className="font-bold tracking-widest uppercase text-xs">No tasks match "{searchQuery}"</p>
                </div>
              ) : filterCategory !== "All" ? (
                <p className="font-bold tracking-widest uppercase text-xs">No tasks in {filterCategory}</p>
              ) : (
                <div className="space-y-4">
                  <Sparkles className="h-12 w-12 mx-auto opacity-20" />
                  <p className="font-bold tracking-widest uppercase text-xs">No tasks yet. Create one above!</p>
                </div>
              )}
            </div>
          ) : (
            filteredTasks?.map((task) => (
              <div key={task.id} className="bg-white dark:bg-white/5 p-6 rounded-[2rem] border border-gray-100 dark:border-white/5 flex flex-col shadow-lg hover:shadow-xl dark:hover:bg-white/[0.08] hover:border-mulberry/20 dark:hover:border-white/10 transition-all duration-500 group">
                {editingId === task.id ? (
                  <div className="space-y-5 p-2">
                    <input
                      className="w-full bg-gray-50 dark:bg-background/50 p-4 border border-gray-200 dark:border-white/10 rounded-2xl focus:ring-2 focus:ring-mulberry outline-none text-lg font-bold transition-all text-gray-900 dark:text-white"
                      value={editTitle}
                      onChange={(e) => setEditTitle(e.target.value)}
                      autoFocus
                    />
                    <textarea
                      className="w-full bg-gray-50 dark:bg-background/50 p-4 border border-gray-200 dark:border-white/10 rounded-2xl focus:ring-2 focus:ring-mulberry outline-none text-sm text-gray-700 dark:text-gray-400 transition-all resize-none"
                      value={editDescription}
                      onChange={(e) => setEditDescription(e.target.value)}
                      rows={3}
                    />
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
                      <div className="flex flex-wrap items-center gap-2">
                        {CATEGORIES.map(cat => (
                          <button
                            key={cat}
                            type="button"
                            onClick={() => setEditCategory(cat)}
                            className={`px-3 py-1.5 rounded-xl text-[10px] font-bold border transition-all duration-300 ${
                              editCategory === cat ? "bg-mulberry text-white border-mulberry" : "bg-white dark:bg-white/5 text-gray-500 border-gray-200 dark:border-white/10"
                            }`}
                          >
                            {cat}
                          </button>
                        ))}
                      </div>
                      <input
                        type="datetime-local"
                        className="bg-gray-50 dark:bg-background/50 p-3 border border-gray-200 dark:border-white/10 rounded-xl text-xs font-bold text-gray-700 dark:text-gray-400 outline-none transition-all"
                        value={editDueDate}
                        onChange={(e) => setEditDueDate(e.target.value)}
                      />
                    </div>
                    <div className="flex justify-end gap-3 pt-5 border-t border-gray-100 dark:border-white/5">
                      <button
                        onClick={() => setEditingId(null)}
                        className="px-6 py-2.5 text-xs font-bold text-gray-500 dark:text-gray-500 dark:hover:text-white transition-colors hover:text-gray-900"
                      >
                        Cancel
                      </button>
                      <button
                        onClick={() => updateTask.mutate({ id: task.id, title: editTitle, description: editDescription, category: editCategory, due_date: editDueDate || undefined })}
                        disabled={updateTask.isPending || !editTitle.trim()}
                        className="px-8 py-2.5 text-xs font-bold bg-mulberry text-white rounded-xl flex items-center gap-2 disabled:opacity-50 hover:bg-mulberry-light shadow-lg shadow-mulberry/20 transition-all"
                      >
                        {updateTask.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
                        Save Changes
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-center justify-between gap-4 md:gap-6">
                    <button onClick={() => toggleComplete.mutate(task.id)} className="shrink-0 transition-transform active:scale-90 group/check">
                      {task.is_completed ? (
                        <div className="h-10 w-10 bg-green-500/10 rounded-2xl flex items-center justify-center border border-green-500/20 shadow-inner">
                          <CheckCircle className="text-green-500 h-6 w-6" />
                        </div>
                      ) : (
                        <div className="h-10 w-10 bg-gray-50 dark:bg-white/5 rounded-2xl flex items-center justify-center border border-gray-200 dark:border-white/10 group-hover/check:border-mulberry/50 transition-all duration-300 shadow-sm">
                           <div className="h-3 w-3 rounded-full bg-gray-300 dark:bg-gray-700 group-hover/check:bg-mulberry transition-all animate-pulse" />
                        </div>
                      )}
                    </button>

                    <div className="flex-1 min-w-0">
                      <div className="flex flex-wrap items-center gap-3 mb-2">
                         <h3 className={`font-bold truncate text-base md:text-lg transition-all duration-500 ${task.is_completed ? "text-gray-400 dark:text-gray-600 line-through opacity-50" : "text-gray-900 dark:text-white group-hover:text-mulberry"}`}>{task.title}</h3>
                         <span className={`px-2 md:px-3 py-1 rounded-lg md:rounded-xl text-[9px] md:text-[10px] font-black uppercase tracking-widest border shrink-0 transition-all duration-500 ${getCategoryColor(task.category)} ${task.is_completed ? "opacity-30 grayscale" : "opacity-100"}`}>
                           {task.category}
                         </span>
                         {task.due_date && !task.is_completed && (
                           <span className={`px-2 md:px-3 py-1 rounded-lg md:rounded-xl text-[9px] md:text-[10px] font-black border flex items-center gap-2 shrink-0 transition-all duration-500 ${
                             isOverdue(task.due_date) ? "bg-red-500/10 text-red-500 dark:text-red-400 border-red-500/20 animate-pulse" : "bg-gray-100 dark:bg-white/5 text-gray-500 dark:text-gray-400 border-gray-200 dark:border-white/10"
                           }`}>
                             <Clock className="h-2.5 w-2.5 md:h-3 md:w-3" /> {formatDueDate(task.due_date)}
                           </span>
                         )}
                      </div>
                      {task.description && (
                        <p className={`text-xs md:text-sm leading-relaxed transition-all duration-500 line-clamp-2 ${task.is_completed ? "text-gray-400 dark:text-gray-700 opacity-40" : "text-gray-600 dark:text-gray-500 group-hover:text-gray-900 dark:group-hover:text-gray-400"}`}>
                          {task.description}
                        </p>
                      )}
                    </div>

                    <div className="flex md:items-center gap-1 md:gap-2 opacity-0 group-hover:opacity-100 transition-all duration-500 transform translate-x-4 group-hover:translate-x-0 shrink-0">
                      <button
                        onClick={() => startEditing(task)}
                        className="text-gray-400 dark:text-gray-500 hover:text-mulberry dark:hover:text-white p-2 md:p-3 rounded-xl md:rounded-2xl bg-white/0 hover:bg-black/5 dark:hover:bg-white/5 border border-transparent hover:border-black/5 dark:hover:border-white/10 transition-all duration-300"
                        title="Edit task"
                      >
                        <Pencil className="h-4 w-4 md:h-5 md:w-5" />
                      </button>
                      <button
                        onClick={() => { if(confirm("Delete this task?")) deleteTask.mutate(task.id); }}
                        className="text-gray-400 dark:text-gray-500 hover:text-red-500 p-2 md:p-3 rounded-xl md:rounded-2xl bg-white/0 hover:bg-red-500/10 border border-transparent hover:border-red-500/20 transition-all duration-300"
                        title="Delete task"
                      >
                        <Trash2 className="h-4 w-4 md:h-5 md:w-5" />
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </main>
    </div>
  );
}
