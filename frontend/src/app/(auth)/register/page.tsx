"use client";

import { useState } from "react";
import { signUp } from "@/lib/auth-client";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Sparkles, Mail, Lock, User, Loader2 } from "lucide-react";

export default function RegisterPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const { error } = await signUp.email({
        email,
        password,
        name,
        callbackURL: "/dashboard",
      });

      if (error) {
        setError(error.message || "Failed to register");
        return;
      }

      router.push("/dashboard");
      router.refresh();
    } catch (err: any) {
      setError(err.message || "Failed to register");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4 py-12 relative overflow-hidden">
      {/* Background Orbs */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none opacity-30">
        <div className="absolute top-[-20%] left-[-10%] w-[60%] h-[60%] bg-mulberry rounded-full blur-[150px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] bg-cocoa rounded-full blur-[150px]" />
      </div>

      <div className="w-full max-w-md z-10">
        <div className="bg-white/5 backdrop-blur-2xl p-10 rounded-[2.5rem] border border-white/10 shadow-2xl space-y-8">
          <div className="text-center space-y-4">
            <div className="inline-flex p-3 bg-cocoa rounded-2xl shadow-xl shadow-cocoa/30 mb-2">
              <Sparkles className="h-8 w-8 text-white" />
            </div>
            <h2 className="text-4xl font-black tracking-tight text-white uppercase italic">
              Sign Up
            </h2>
            <p className="text-gray-500 font-medium tracking-wide">Start managing your tasks better.</p>
          </div>

          <form className="space-y-6" onSubmit={handleSubmit}>
            {error && (
              <div className="rounded-2xl bg-red-500/10 border border-red-500/20 p-4 text-xs font-bold text-red-400 uppercase tracking-widest text-center">
                {error}
              </div>
            )}

            <div className="space-y-4">
              <div className="relative group">
                <User className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-600 group-focus-within:text-mulberry transition-colors" />
                <input
                  type="text"
                  required
                  className="w-full bg-background/50 pl-12 pr-6 py-4 border border-white/5 rounded-2xl text-white placeholder:text-gray-600 focus:ring-2 focus:ring-cocoa outline-none transition-all"
                  placeholder="Full Name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                />
              </div>

              <div className="relative group">
                <Mail className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-600 group-focus-within:text-mulberry transition-colors" />
                <input
                  type="email"
                  required
                  className="w-full bg-background/50 pl-12 pr-6 py-4 border border-white/5 rounded-2xl text-white placeholder:text-gray-600 focus:ring-2 focus:ring-cocoa outline-none transition-all"
                  placeholder="Email Address"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />
              </div>

              <div className="relative group">
                <Lock className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-600 group-focus-within:text-mulberry transition-colors" />
                <input
                  type="password"
                  required
                  className="w-full bg-background/50 pl-12 pr-6 py-4 border border-white/5 rounded-2xl text-white placeholder:text-gray-600 focus:ring-2 focus:ring-cocoa outline-none transition-all"
                  placeholder="Password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-cocoa text-white py-5 rounded-2xl font-black uppercase tracking-[0.2em] hover:bg-cocoa-light disabled:opacity-50 shadow-xl shadow-cocoa/20 transition-all duration-300 hover:scale-[1.02] active:scale-[0.98] flex items-center justify-center gap-3"
            >
              {loading ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : (
                "Create Account"
              )}
            </button>
          </form>

          <div className="text-center pt-4">
            <p className="text-sm text-gray-600 font-medium">
              Already have an account?{" "}
              <Link href="/login" className="text-cocoa-light hover:text-white transition-colors underline decoration-cocoa/30 underline-offset-4">
                Login Here
              </Link>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
