import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Toaster, toast } from 'sonner';
import { LoginForm } from './components/LoginForm';
import { AnimatedSphere } from './components/AnimatedSphere';
import { Dashboard } from './components/Dashboard';
import { loginUser, registerUser } from './api/api';

interface User {
  id: string;
  email: string;
}

export default function App() {
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [isDark, setIsDark] = useState(true);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', isDark);
  }, [isDark]);

  const handleLogin = async (email: string, password: string) => {
    try {
      const user = await loginUser(email, password);
      setIsTransitioning(true);
      setTimeout(() => {
        setCurrentUser(user);
      }, 2100);
    } catch (error: any) {
      toast.error(error.message || 'Не удалось войти');
    }
  };

  const handleRegister = async (email: string, password: string) => {
    try {
      await registerUser(email, password);
      toast.success('Регистрация успешна! Теперь вы можете войти.');
    } catch (error: any) {
      toast.error(error.message || 'Не удалось зарегистрироваться');
    }
  };

  const handleLogout = () => {
    setCurrentUser(null);
    setIsTransitioning(false);
  };

  if (currentUser) {
    return (
      <>
        <Dashboard
          user={currentUser}
          onLogout={handleLogout}
          isDark={isDark}
          onThemeToggle={() => setIsDark(!isDark)}
        />
        <Toaster position="top-right" theme={isDark ? 'dark' : 'light'} richColors />
      </>
    );
  }

  return (
    <div className={`min-h-screen w-full relative overflow-hidden transition-colors duration-500 ${
      isDark 
        ? 'bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900' 
        : 'bg-gradient-to-br from-blue-50 via-purple-50 to-cyan-50'
    }`}>
      <AnimatedSphere isExpanding={isTransitioning} />
      <AnimatePresence>
        {!isTransitioning && (
          <motion.div
            className="absolute inset-0 flex items-center justify-center z-20"
            initial={{ opacity: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            transition={{ duration: 0.3 }}
          >
            <div className="w-full max-w-md px-6">
              <LoginForm
                onLogin={handleLogin}
                onRegister={handleRegister}
                isDark={isDark}
                onThemeToggle={() => setIsDark(!isDark)}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      <Toaster position="top-right" theme={isDark ? 'dark' : 'light'} richColors />
    </div>
  );
}