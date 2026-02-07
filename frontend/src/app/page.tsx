"use client";

import Link from "next/link";
import { Sparkles, ArrowRight, CheckCircle2, Shield, Zap, Globe } from "lucide-react";
import { useEffect, useState } from "react";

export default function Home() {
  const [isDarkMode, setIsDarkMode] = useState(true);

  useEffect(() => {
    // Initialize theme from local storage if on client
    const savedTheme = localStorage.getItem("theme");
    if (savedTheme === "light") {
        setIsDarkMode(false);
        document.documentElement.classList.remove("dark");
    } else {
        setIsDarkMode(true);
        document.documentElement.classList.add("dark");
    }
  }, []);

  return (
    <div className="min-h-screen bg-background text-foreground transition-colors duration-500 relative overflow-hidden flex flex-col">
      {/* Background Orbs */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none opacity-20 dark:opacity-30">
        <div className="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] bg-mulberry rounded-full blur-[150px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] bg-cocoa rounded-full blur-[150px]" />
      </div>

      {/* Navigation */}
      <nav className="z-50 px-8 py-6 flex justify-between items-center max-w-7xl mx-auto w-full">
        <div className="flex items-center gap-2">
            <div className="p-2 bg-mulberry rounded-xl shadow-lg shadow-mulberry/20">
              <Sparkles className="h-5 w-5 text-white" />
            </div>
            <span className="text-xl font-bold tracking-tighter uppercase italic dark:text-white">Elevate</span>
        </div>
        <div className="flex items-center gap-8">
            <Link href="/login" className="text-sm font-bold uppercase tracking-widest text-gray-500 hover:text-mulberry transition-colors">Login</Link>
            <Link href="/tasks" className="text-sm font-bold uppercase tracking-widest text-gray-500 hover:text-mulberry transition-colors">Tasks</Link>
            <Link href="/register" className="bg-white/5 dark:bg-white/10 backdrop-blur-md border border-black/5 dark:border-white/10 px-6 py-2.5 rounded-2xl text-sm font-bold uppercase tracking-widest hover:bg-mulberry hover:text-white transition-all duration-300">Join Nexus</Link>
        </div>
      </nav>

      {/* Hero Section */}
      <main className="flex-1 flex flex-col items-center justify-center px-6 relative z-10 py-20">
        <div className="max-w-4xl text-center space-y-12">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-white/5 backdrop-blur-md border border-black/5 dark:border-white/10 rounded-full animate-bounce">
            <Zap className="h-4 w-4 text-mulberry-light" />
            <span className="text-[10px] font-black uppercase tracking-[0.3em] text-gray-500 dark:text-gray-400">Release 2.0 is Live</span>
          </div>

          <div className="space-y-6">
            <h1 className="text-6xl md:text-8xl font-black tracking-tight leading-[0.9] text-gray-900 dark:text-white transition-colors">
              ASCEND YOUR <br />
              <span className="bg-gradient-to-r from-mulberry via-mulberry-light to-cocoa bg-clip-text text-transparent">PRODUCTIVITY.</span>
            </h1>
            <p className="max-w-2xl mx-auto text-lg md:text-xl text-gray-500 dark:text-gray-400 font-medium leading-relaxed">
              Experience the pinnacle of task management. A premium workspace designed for those who demand focus and aesthetic perfection.
            </p>
          </div>

          <div className="flex flex-col sm:flex-row gap-6 justify-center pt-4">
            <Link
              href="/tasks"
              className="group bg-mulberry text-white px-10 py-5 rounded-[2rem] text-lg font-black uppercase tracking-[0.2em] shadow-2xl shadow-mulberry/30 hover:bg-mulberry-light hover:scale-105 active:scale-95 transition-all duration-300 flex items-center justify-center gap-3"
            >
              Manage Tasks <ArrowRight className="h-5 w-5 group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link
              href="/login"
              className="bg-white/5 dark:bg-white/5 backdrop-blur-xl px-10 py-5 rounded-[2rem] text-lg font-black uppercase tracking-[0.2em] border border-black/5 dark:border-white/10 hover:bg-black/10 dark:hover:bg-white/10 transition-all duration-300 shadow-xl"
            >
              Enter Nexus
            </Link>
          </div>

          {/* Features Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-32 text-left">
            <div className="p-8 bg-white/5 dark:bg-white/5 backdrop-blur-md rounded-[2.5rem] border border-black/5 dark:border-white/10 space-y-4 hover:border-mulberry/30 transition-all duration-500 group">
                <div className="p-3 bg-mulberry/10 rounded-2xl w-fit text-mulberry group-hover:scale-110 transition-transform">
                    <Shield className="h-6 w-6" />
                </div>
                <h3 className="text-xl font-bold dark:text-white italic">Encrypted Core</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400 leading-relaxed">Your data is secured with AES-256 standard encryption, ensuring your focus remains private.</p>
            </div>

            <div className="p-8 bg-white/5 dark:bg-white/5 backdrop-blur-md rounded-[2.5rem] border border-black/5 dark:border-white/10 space-y-4 hover:border-mulberry/30 transition-all duration-500 group">
                <div className="p-3 bg-cocoa/10 rounded-2xl w-fit text-cocoa-light group-hover:scale-110 transition-transform">
                    <Zap className="h-6 w-6" />
                </div>
                <h3 className="text-xl font-bold dark:text-white italic">Peak Velocity</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400 leading-relaxed">Lightning-fast synchronization across all your dimensions and devices instantly.</p>
            </div>

            <div className="p-8 bg-white/5 dark:bg-white/5 backdrop-blur-md rounded-[2.5rem] border border-black/5 dark:border-white/10 space-y-4 hover:border-mulberry/30 transition-all duration-500 group">
                <div className="p-3 bg-mulberry/10 rounded-2xl w-fit text-mulberry-light group-hover:scale-110 transition-transform">
                    <Globe className="h-6 w-6" />
                </div>
                <h3 className="text-xl font-bold dark:text-white italic">Global Nexus</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400 leading-relaxed">Access your tasks from anywhere in the world with our global edge network.</p>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="z-10 px-8 py-10 border-t border-black/5 dark:border-white/5 bg-background/50 backdrop-blur-md mt-20">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-6">
            <div className="flex items-center gap-2">
                <div className="p-1 px-2 bg-mulberry/10 rounded-lg">
                    <Sparkles className="h-4 w-4 text-mulberry" />
                </div>
                <span className="text-xs font-black uppercase tracking-[0.4em] text-gray-900 dark:text-white">Elevate Nexus</span>
            </div>
            <p className="text-[10px] font-black uppercase tracking-[0.2em] text-gray-500">
                Created to redefine the standard of productivity systems.
            </p>
            <div className="text-[10px] font-black uppercase tracking-[0.1em] text-gray-600">
                © 2026 ELEVATE. DISCIPLINE IS FREEDOM.
            </div>
        </div>
      </footer>
    </div>
  );
}
