import { motion } from 'motion/react';
import { useEffect, useState } from 'react';

interface AnimatedSphereProps {
  isExpanding: boolean;
}

export function AnimatedSphere({ isExpanding }: AnimatedSphereProps) {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({
        x: (e.clientX / window.innerWidth - 0.5) * 30,
        y: (e.clientY / window.innerHeight - 0.5) * 30
      });
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  return (
    <div className="absolute inset-0 flex items-center justify-center overflow-hidden pointer-events-none">
      {/* Основная сфера */}
      <motion.div
        className="relative"
        initial={{ scale: 0.4, opacity: 0 }}
        animate={
          isExpanding 
            ? { scale: 15, opacity: 0.6 }
            : { 
                scale: [0.4, 1.3],
                opacity: [0, 0.7],
                x: mousePosition.x * 0.4,
                y: mousePosition.y * 0.4
              }
        }
        transition={
          isExpanding 
            ? { duration: 2, ease: "easeInOut" }
            : { 
                scale: { duration: 2, ease: "easeOut" },
                opacity: { duration: 1.5, ease: "easeOut" },
                x: { duration: 2.5, ease: "easeOut" },
                y: { duration: 2.5, ease: "easeOut" }
              }
        }
      >
        {/* Основная деформируемая сфера - увеличена с 96 до 160 */}
        <motion.div 
          className="w-[640px] h-[640px] rounded-full relative"
          animate={{
            scaleX: [1, 1.12, 0.88, 1.06, 0.94, 1],
            scaleY: [1, 0.88, 1.12, 0.94, 1.06, 1],
            x: [0, 30, -18, 25, -12, 0],
            y: [0, -15, 22, -8, 16, 0],
            rotate: [0, 4, -3, 5, -2, 0]
          }}
          transition={{
            duration: 8,
            ease: "easeInOut",
            times: [0, 0.2, 0.4, 0.6, 0.8, 1],
            repeat: Infinity
          }}
        >
          {/* Основной градиент - менее яркий */}
          <motion.div 
            className="absolute inset-0 rounded-full bg-gradient-to-br from-purple-600/15 via-blue-500/10 to-cyan-400/15 blur-2xl"
            animate={{
              scaleX: [1, 0.85, 1.15, 0.92, 1],
              scaleY: [1, 1.15, 0.85, 1.08, 1],
              opacity: [0.15, 0.25, 0.10, 0.20, 0.15]
            }}
            transition={{
              duration: 6,
              ease: "easeInOut",
              times: [0, 0.3, 0.6, 0.8, 1],
              repeat: Infinity
            }}
          />
          
          {/* Внутренний деформируемый слой */}
          <motion.div 
            className="absolute inset-12 rounded-full bg-gradient-to-br from-purple-500/20 via-blue-600/15 to-cyan-500/20 blur-xl"
            animate={{
              scaleX: [1, 1.25, 0.75, 1.15, 0.85, 1],
              scaleY: [1, 0.75, 1.25, 0.85, 1.15, 1],
              x: [0, -12, 18, -8, 14, 0],
              y: [0, 16, -10, 12, -6, 0],
              rotate: [0, -3, 5, -2, 4, 0]
            }}
            transition={{
              duration: 7,
              ease: "easeInOut",
              times: [0, 0.2, 0.4, 0.6, 0.8, 1],
              repeat: Infinity
            }}
          />
          
          {/* Центральное ядро с органичными деформациями */}
          <motion.div 
            className="absolute inset-24 rounded-full bg-gradient-to-br from-purple-400/25 via-blue-500/20 to-cyan-400/25 blur-lg"
            animate={{
              scaleX: [1, 0.8, 1.3, 0.9, 1.2, 1],
              scaleY: [1, 1.3, 0.7, 1.1, 0.8, 1],
              rotate: [0, 8, -6, 4, -3, 0],
              x: [0, 8, -12, 5, -3, 0],
              y: [0, -10, 6, -4, 8, 0]
            }}
            transition={{
              duration: 9,
              ease: "easeInOut",
              times: [0, 0.25, 0.5, 0.7, 0.9, 1],
              repeat: Infinity
            }}
          />
        </motion.div>

        {/* Более активные органичные блики */}
        <motion.div
          className="absolute top-24 left-24 w-56 h-56 rounded-full bg-white/8 blur-2xl"
          animate={{
            x: [0, 100, -50, 80, -30, 0],
            y: [0, -35, 70, -20, 45, 0],
            scaleX: [1, 1.6, 0.5, 1.4, 0.7, 1],
            scaleY: [1, 0.6, 1.8, 0.8, 1.3, 1],
            opacity: [0.08, 0.2, 0.12, 0.18, 0.08],
            rotate: [0, 20, -15, 12, -8, 0]
          }}
          transition={{
            duration: 10,
            ease: "easeInOut",
            times: [0, 0.25, 0.5, 0.75, 0.9, 1],
            repeat: Infinity
          }}
        />

        <motion.div
          className="absolute bottom-24 right-24 w-44 h-44 rounded-full bg-cyan-300/12 blur-xl"
          animate={{
            x: [0, -90, 40, -60, 25, 0],
            y: [0, 50, -60, 40, -20, 0],
            scaleX: [1, 0.4, 1.9, 0.6, 1.4, 1],
            scaleY: [1, 1.7, 0.5, 1.3, 0.7, 1],
            rotate: [0, 25, -20, 15, -10, 0],
            opacity: [0.12, 0.3, 0.08, 0.25, 0.12]
          }}
          transition={{
            duration: 8,
            ease: "easeInOut",
            times: [0, 0.3, 0.55, 0.8, 1],
            repeat: Infinity
          }}
        />

        {/* Дополнительные более активные органичные элементы */}
        <motion.div
          className="absolute top-48 right-48 w-32 h-32 rounded-full bg-purple-200/15 blur-lg"
          animate={{
            x: [0, 45, -60, 20, -12, 0],
            y: [0, -40, 65, -25, 35, 0],
            scaleX: [1, 2.5, 0.2, 1.8, 0.6, 1],
            scaleY: [1, 0.3, 2.1, 0.5, 1.6, 1],
            opacity: [0.15, 0.35, 0.08, 0.28, 0.15],
            rotate: [0, 35, -25, 15, -8, 0]
          }}
          transition={{
            duration: 7,
            ease: "easeInOut",
            times: [0, 0.2, 0.45, 0.7, 0.9, 1],
            repeat: Infinity
          }}
        />

        {/* Более активные плавающие частицы */}
        <motion.div
          className="absolute top-60 left-60 w-16 h-16 rounded-full bg-blue-300/20 blur-md"
          animate={{
            x: [0, 70, -40, 50, -25, 0],
            y: [0, -45, 55, -20, 30, 0],
            scale: [1, 3, 0.3, 2.2, 0.5, 1],
            opacity: [0.2, 0.4, 0.08, 0.35, 0.2],
            rotate: [0, 60, -40, 25, -20, 0]
          }}
          transition={{
            duration: 6,
            ease: "easeInOut",
            times: [0, 0.3, 0.6, 0.8, 1],
            repeat: Infinity
          }}
        />

        {/* Дополнительные активные частицы */}
        <motion.div
          className="absolute bottom-48 left-48 w-20 h-20 rounded-full bg-purple-300/15 blur-lg"
          animate={{
            x: [0, -50, 35, -40, 15, 0],
            y: [0, 45, -35, 30, -20, 0],
            scale: [1, 2.2, 0.2, 1.7, 0.6, 1],
            opacity: [0.15, 0.3, 0.04, 0.25, 0.15],
            rotate: [0, -30, 45, -18, 12, 0]
          }}
          transition={{
            duration: 5,
            ease: "easeInOut",
            times: [0, 0.25, 0.5, 0.75, 1],
            repeat: Infinity
          }}
        />
      </motion.div>

      {/* Внешние дышащие кольца - менее яркие */}
      <motion.div
        className="absolute w-[800px] h-[800px] rounded-full border border-purple-400/4"
        animate={{ 
          scaleX: [1, 1.08, 0.92, 1.05, 0.95, 1],
          scaleY: [1, 0.92, 1.08, 0.95, 1.05, 1],
          opacity: [0.04, 0.12, 0.02, 0.08, 0.04],
          rotate: [0, 3, -2, 4, -1, 0]
        }}
        transition={{ 
          duration: 12,
          ease: "easeInOut",
          times: [0, 0.2, 0.4, 0.6, 0.8, 1],
          repeat: Infinity
        }}
      />

      <motion.div
        className="absolute w-[700px] h-[700px] rounded-full border border-cyan-300/6"
        animate={{ 
          scaleX: [1, 0.88, 1.12, 0.93, 1.07, 1],
          scaleY: [1, 1.12, 0.88, 1.07, 0.93, 1],
          opacity: [0.06, 0.15, 0.02, 0.12, 0.06],
          rotate: [0, -4, 3, -2, 5, 0]
        }}
        transition={{ 
          duration: 10,
          ease: "easeInOut",
          times: [0, 0.3, 0.6, 0.8, 1],
          repeat: Infinity
        }}
      />
    </div>
  );
}