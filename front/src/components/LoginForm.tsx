import { useState } from 'react';
import { motion } from 'motion/react';
import { GlassButton } from './GlassButton';
import { ThemeToggle } from './ThemeToggle';
import { Input } from './ui/input';
import { Label } from './ui/label';

interface LoginFormProps {
  onLogin: (email: string, password: string) => void;
  onRegister: (email: string, password: string) => void;
  isDark: boolean;
  onThemeToggle: () => void;
}

export function LoginForm({ onLogin, onRegister, isDark, onThemeToggle }: LoginFormProps) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isRegisterMode, setIsRegisterMode] = useState(false);

  // Микроанимации для полей
  const fieldVariants = {
    hidden: { opacity: 0, y: 20, scale: 0.95 },
    visible: (custom: number) => ({
      opacity: 1,
      y: 0,
      scale: 1,
      transition: {
        delay: custom * 0.1,
        duration: 0.6,
        ease: [0.25, 0.46, 0.45, 0.94]
      }
    })
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.2
      }
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (isRegisterMode) {
      onRegister(email, password);
      setEmail('');
      setPassword('');
      setIsRegisterMode(false);
    } else {
      onLogin(email, password);
    }
  };

  return (
    <motion.div
      initial="hidden"
      animate="visible"
      variants={containerVariants}
      className={`backdrop-blur-xl p-8 rounded-3xl shadow-2xl border transition-all duration-500 ${
        isDark 
          ? 'bg-white/[0.08] border-white/20 shadow-black/20' 
          : 'bg-white/[0.15] border-white/30 shadow-black/10'
      }`}
      style={{
        boxShadow: isDark
          ? '0 25px 50px -12px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1)'
          : '0 25px 50px -12px rgba(0, 0, 0, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.2)'
      }}
    >
      <div className="flex justify-between items-start mb-8">
        <motion.div 
          className="flex-1 text-left"
          variants={fieldVariants}
          custom={0}
        >
          <motion.h1 
            className={`mb-2 ${isDark ? 'text-white' : 'text-slate-900'}`}
            whileHover={{ scale: 1.02 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
          >
            Медицинская система МРТ
          </motion.h1>
          {isRegisterMode && (
            <motion.p 
              className={`${isDark ? 'text-white/70' : 'text-slate-700'}`}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5, duration: 0.8 }}
            >
              Регистрация в системе
            </motion.p>
          )}
        </motion.div>
        <motion.div 
          className="ml-4"
          variants={fieldVariants}
          custom={1}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.95 }}
        >
          <ThemeToggle isDark={isDark} onToggle={onThemeToggle} />
        </motion.div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <motion.div 
          className="space-y-2"
          variants={fieldVariants}
          custom={2}
        >
          <Label 
            htmlFor="email" 
            className={`transition-colors duration-300 ${isDark ? 'text-white' : 'text-slate-900'}`}
          >
            Email
          </Label>
          <motion.div
            whileFocus={{ scale: 1.02 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
          >
            <Input
              id="email"
              type="email"
              placeholder="doctor@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className={`transition-all duration-300 rounded-xl border-0 backdrop-blur-sm ${
                isDark 
                  ? 'bg-white/10 border-white/20 text-white placeholder:text-white/50 focus:bg-white/15 shadow-inner' 
                  : 'bg-white/20 border-slate-200/30 text-slate-900 placeholder:text-slate-500 focus:bg-white/30 shadow-inner'
              }`}
              style={{
                boxShadow: isDark
                  ? 'inset 0 2px 8px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(255, 255, 255, 0.1)'
                  : 'inset 0 2px 8px rgba(0, 0, 0, 0.05), 0 1px 2px rgba(255, 255, 255, 0.2)'
              }}
              required
            />
          </motion.div>
        </motion.div>

        <motion.div 
          className="space-y-2"
          variants={fieldVariants}
          custom={3}
        >
          <Label 
            htmlFor="password" 
            className={`transition-colors duration-300 ${isDark ? 'text-white' : 'text-slate-900'}`}
          >
            Пароль
          </Label>
          <motion.div
            whileFocus={{ scale: 1.02 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
          >
            <Input
              id="password"
              type="password"
              placeholder="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className={`transition-all duration-300 rounded-xl border-0 backdrop-blur-sm ${
                isDark 
                  ? 'bg-white/10 border-white/20 text-white placeholder:text-white/50 focus:bg-white/15 shadow-inner' 
                  : 'bg-white/20 border-slate-200/30 text-slate-900 placeholder:text-slate-500 focus:bg-white/30 shadow-inner'
              }`}
              style={{
                boxShadow: isDark
                  ? 'inset 0 2px 8px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(255, 255, 255, 0.1)'
                  : 'inset 0 2px 8px rgba(0, 0, 0, 0.05), 0 1px 2px rgba(255, 255, 255, 0.2)'
              }}
              required
            />
          </motion.div>
        </motion.div>

        <motion.div
          variants={fieldVariants}
          custom={4}
        >
          <GlassButton 
            className="w-full mt-6" 
            variant="primary"
            size="lg"
          >
            <motion.span
              whileHover={{ letterSpacing: "0.02em" }}
              transition={{ duration: 0.2 }}
            >
              {isRegisterMode ? 'Зарегистрироваться' : 'Войти в систему'}
            </motion.span>
          </GlassButton>
        </motion.div>

        <motion.div 
          className="flex justify-center mt-4"
          variants={fieldVariants}
          custom={5}
        >
          <motion.button
            type="button"
            onClick={() => setIsRegisterMode(!isRegisterMode)}
            className={`text-sm hover:underline transition-colors ${
              isDark ? 'text-white/70 hover:text-white' : 'text-slate-600 hover:text-slate-900'
            }`}
            whileHover={{ 
              scale: 1.05,
              transition: { type: "spring", stiffness: 300, damping: 30 }
            }}
            whileTap={{ scale: 0.95 }}
          >
            {isRegisterMode ? 'Уже есть аккаунт? Войти' : 'Нет аккаунта? Зарегистрироваться'}
          </motion.button>
        </motion.div>

        <motion.p 
          className={`text-center text-sm mt-4 transition-colors duration-300 ${
            isDark ? 'text-white/50' : 'text-slate-600'
          }`}
          variants={fieldVariants}
          custom={6}
          whileHover={{ 
            scale: 1.05,
            transition: { type: "spring", stiffness: 300, damping: 30 }
          }}
        >
          {isRegisterMode 
            ? 'Создайте новый аккаунт для входа в систему' 
            : 'Войдите или зарегистрируйтесь для доступа'
          }
        </motion.p>
      </form>
    </motion.div>
  );
}