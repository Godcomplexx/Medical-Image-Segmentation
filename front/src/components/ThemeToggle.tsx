import { useState, useEffect } from 'react';
import { motion } from 'motion/react';
import { Sun, Moon } from 'lucide-react';

interface ThemeToggleProps {
  isDark: boolean;
  onToggle: () => void;
}

export function ThemeToggle({ isDark, onToggle }: ThemeToggleProps) {
  return (
    <motion.button
      onClick={onToggle}
      className="relative flex items-center w-14 h-7 bg-white/10 backdrop-blur-sm border border-white/20 rounded-full p-1 cursor-pointer"
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
    >
      {/* Фоновый трек */}
      <motion.div
        className="absolute inset-1 rounded-full"
        animate={{
          background: isDark 
            ? 'linear-gradient(90deg, #1e293b, #334155)' 
            : 'linear-gradient(90deg, #fbbf24, #f59e0b)'
        }}
        transition={{ duration: 0.3 }}
      />
      
      {/* Скользящий элемент */}
      <motion.div
        className="relative w-5 h-5 bg-white rounded-full shadow-lg flex items-center justify-center z-10"
        animate={{
          x: isDark ? 24 : 0,
        }}
        transition={{
          type: "spring",
          stiffness: 500,
          damping: 30
        }}
      >
        <motion.div
          animate={{
            rotate: isDark ? 180 : 0,
            scale: isDark ? 0.8 : 1
          }}
          transition={{ duration: 0.3 }}
        >
          {isDark ? (
            <Moon className="w-3 h-3 text-slate-700" />
          ) : (
            <Sun className="w-3 h-3 text-yellow-600" />
          )}
        </motion.div>
      </motion.div>

      {/* Анимированные иконки в фоне */}
      <div className="absolute inset-0 flex items-center justify-between px-2">
        <motion.div
          animate={{
            opacity: isDark ? 0 : 0.5,
            scale: isDark ? 0.8 : 1
          }}
          transition={{ duration: 0.3 }}
        >
          <Sun className="w-3 h-3 text-white" />
        </motion.div>
        <motion.div
          animate={{
            opacity: isDark ? 0.5 : 0,
            scale: isDark ? 1 : 0.8
          }}
          transition={{ duration: 0.3 }}
        >
          <Moon className="w-3 h-3 text-white" />
        </motion.div>
      </div>
    </motion.button>
  );
}