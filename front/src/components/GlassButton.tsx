import React from 'react';
import { motion } from 'motion/react';
import { ReactNode, ButtonHTMLAttributes, MouseEventHandler } from 'react';

interface GlassButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  children?: ReactNode;
  variant?: 'primary' | 'secondary' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
  disabled?: boolean;
  onClick?: MouseEventHandler<HTMLButtonElement>;
  type?: 'button' | 'submit' | 'reset';
}

export function GlassButton({
  children,
  variant = 'primary',
  size = 'md',
  className = '',
  disabled = false,
  ...rest
}: GlassButtonProps) {
  const sizeClasses = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-4 py-2',
    lg: 'px-6 py-3'
  };

  const variantClasses = {
    primary: 'bg-gradient-to-r from-blue-600 to-purple-600 border-blue-500/50 text-white shadow-lg shadow-blue-500/25 hover:from-blue-700 hover:to-purple-700 dark:bg-white/10 dark:border-white/20 dark:hover:bg-white/20 dark:text-white dark:from-white/10 dark:to-white/10',
    secondary: 'bg-slate-900/10 border-slate-900/20 text-slate-900 shadow-lg shadow-slate-900/10 hover:bg-slate-900/20 dark:bg-blue-500/10 dark:border-blue-300/20 dark:text-blue-100 dark:hover:bg-blue-500/20 dark:shadow-blue-500/10',
    ghost: 'bg-slate-900/5 border-slate-900/10 text-slate-900 shadow-md shadow-slate-900/5 hover:bg-slate-900/10 dark:bg-white/5 dark:border-white/10 dark:text-white/70 dark:hover:bg-white/10 dark:shadow-white/5'
  };

  return (
    <motion.button
      disabled={disabled}
      className={`
        relative overflow-hidden backdrop-blur-sm border rounded-lg
        transition-all duration-200 font-medium
        disabled:opacity-50 disabled:cursor-not-allowed
        ${sizeClasses[size]}
        ${variantClasses[variant]}
        ${className}
      `}
      {...rest}
      whileHover={!disabled ? {
        y: -2,
        boxShadow: variant === 'primary'
          ? '0 12px 40px rgba(0, 0, 0, 0.18)'
          : '0 10px 28px rgba(0, 0, 0, 0.12)',
        transition: { duration: 0.14, ease: [0.2, 0.7, 0.1, 1] }
      } : {}}
      whileTap={!disabled ? {
        scale: 0.985,
        boxShadow: '0 6px 16px rgba(0,0,0,0.10)',
        transition: { duration: 0.09, ease: 'easeOut' }
      } : {}}
    >
      {/* Hover эффект */}
      <motion.div
        className="absolute inset-0 bg-white/10 opacity-0 hover:opacity-100 dark:bg-white/5"
        transition={{ duration: 0.2 }}
      />
      
      {/* Контент кнопки */}
      <span className="relative z-10 inline-flex items-center">
        {children}
      </span>
    </motion.button>
  );
}