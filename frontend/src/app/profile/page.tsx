"use client";

import { useSession } from "@/lib/auth-client";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { User, Mail, Calendar, Settings, ArrowLeft, Loader2, Sparkles, ShieldCheck } from "lucide-react";
import Link from "next/link";

export default function ProfilePage() {
  const { data: session, isPending } = useSession();
  const router = useRouter();

  useEffect(() => {
    if (!isPending && !session) {
      router.push("/login");
    }
  }, [session, isPending, router]);

  if (isPending || !session) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <Loader2 className="h-10 w-10 animate-spin text-mulberry" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background text-foreground pb-20 relative overflow-hidden">
        {/* Background Orbs */}
        <div className="fixed inset-0 overflow-hidden pointer-events-none opacity-20">
            <div className="absolute top-[-10%] right-[-10%] w-[50%] h-[50%] bg-mulberry rounded-full blur-[120px]" />
            <div className="absolute bottom-[-10%] left-[-10%] w-[50%] h-[50%] bg-cocoa rounded-full blur-[120px]" />
        </div>

      <nav className="border-b border-white/5 mb-12 backdrop-blur-xl bg-background/80 sticky top-0 z-20">
        <div className="mx-auto max-w-5xl px-8 items-center flex h-20">
            <Link
              href="/dashboard"
              className="flex items-center gap-3 text-sm font-bold uppercase tracking-widest text-gray-400 hover:text-white transition-all group"
            >
              <ArrowLeft className="h-5 w-5 group-hover:-translate-x-1 transition-transform" />
              Return to Dashboard
            </Link>
        </div>
      </nav>

      <main className="mx-auto max-w-3xl px-8 z-10 relative">
        <div className="bg-white/5 backdrop-blur-2xl rounded-[3rem] border border-white/10 overflow-hidden shadow-2xl">
          <div className="bg-gradient-to-br from-mulberry to-cocoa h-48 relative">
             <div className="absolute inset-0 bg-black/20 backdrop-blur-[2px]"></div>
             <div className="absolute bottom-6 left-10 flex items-center gap-4">
                 <div className="p-4 bg-white/10 backdrop-blur-md rounded-2xl border border-white/20">
                    <Sparkles className="h-8 w-8 text-white" />
                 </div>
                 <div>
                    <h1 className="text-3xl font-black text-white uppercase italic tracking-tighter">My Profile</h1>
                    <p className="text-white/60 text-xs font-bold uppercase tracking-[0.3em]">Account Verified</p>
                 </div>
             </div>
          </div>
          <div className="px-10 pb-12 pt-10">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-10">
              <div className="flex items-center gap-5 p-6 rounded-3xl bg-background/50 border border-white/5 group hover:border-mulberry/30 transition-all duration-500">
                <div className="p-3 bg-white/5 rounded-2xl text-gray-400 group-hover:text-mulberry-light transition-colors">
                    <User className="h-6 w-6" />
                </div>
                <div>
                  <p className="text-[10px] font-black text-gray-500 uppercase tracking-[0.2em] mb-1">Full Name</p>
                  <p className="text-lg font-bold text-white tracking-tight">{session.user.name}</p>
                </div>
              </div>

              <div className="flex items-center gap-5 p-6 rounded-3xl bg-background/50 border border-white/5 group hover:border-mulberry/30 transition-all duration-500">
                <div className="p-3 bg-white/5 rounded-2xl text-gray-400 group-hover:text-mulberry-light transition-colors">
                    <Mail className="h-6 w-6" />
                </div>
                <div>
                  <p className="text-[10px] font-black text-gray-500 uppercase tracking-[0.2em] mb-1">Email Address</p>
                  <p className="text-lg font-bold text-white tracking-tight truncate max-w-[180px]">{session.user.email}</p>
                </div>
              </div>

              <div className="flex items-center gap-5 p-6 rounded-3xl bg-background/50 border border-white/5 group hover:border-mulberry/30 transition-all duration-500 md:col-span-2">
                <div className="p-3 bg-white/5 rounded-2xl text-gray-400 group-hover:text-mulberry-light transition-colors">
                    <Calendar className="h-6 w-6" />
                </div>
                <div>
                  <p className="text-[10px] font-black text-gray-500 uppercase tracking-[0.2em] mb-1">Member Since</p>
                  <p className="text-lg font-bold text-white tracking-tight">
                    {new Date(session.user.createdAt).toLocaleDateString(undefined, {
                      year: 'numeric',
                      month: 'long',
                      day: 'numeric'
                    })}
                  </p>
                </div>
              </div>
            </div>

            <div className="pt-8 border-t border-white/5 flex flex-col items-center gap-6">
                <div className="flex items-center gap-2 text-gray-600">
                    <ShieldCheck className="h-4 w-4" />
                    <span className="text-[10px] uppercase font-black tracking-widest">Secure Profile</span>
                </div>
                <button
                  disabled
                  className="flex items-center justify-center gap-3 w-full rounded-2xl bg-white/5 px-6 py-5 text-sm font-black uppercase tracking-[0.2em] text-gray-600 border border-white/5 cursor-not-allowed opacity-50"
                >
                  <Settings className="h-5 w-5" />
                  Edit Settings (Soon)
                </button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
