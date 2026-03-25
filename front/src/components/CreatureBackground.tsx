import { useEffect, useRef } from 'react';

interface CreatureBackgroundProps {
  isDark: boolean;
}

export function CreatureBackground({ isDark }: CreatureBackgroundProps) {
  const creatureRef = useRef<HTMLDivElement>(null);
  const animationRef = useRef<number | null>(null);
  const cursorRef = useRef({ x: 0, y: 0 });
  const isAutoMovingRef = useRef(true);
  const manualTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (!creatureRef.current) return;

    const creatureEl = creatureRef.current;
    const viewport = { w: window.innerWidth * 0.5, h: window.innerHeight * 0.5 };
    const cursor = cursorRef.current;
    const rows = 13;
    const grid = [rows, rows];
    let currentTime = 0;

    // Создаем элементы
    creatureEl.innerHTML = '';
    for (let i = 0; i < (rows * rows); i++) {
      const div = document.createElement('div');
      div.className = 'creature-particle';
      creatureEl.appendChild(div);
    }

    const particleEls = creatureEl.querySelectorAll('.creature-particle') as NodeListOf<HTMLElement>;

    // Настройка стилей контейнера
    creatureEl.style.width = rows * 10 + 'em';
    creatureEl.style.height = rows * 10 + 'em';

    // Инициализация частиц
    particleEls.forEach((el, i) => {
      const row = Math.floor(i / rows);
      const col = i % rows;
      const centerRow = Math.floor(rows / 2);
      const centerCol = Math.floor(rows / 2);
      const distanceFromCenter = Math.sqrt(Math.pow(row - centerRow, 2) + Math.pow(col - centerCol, 2));
      const maxDistance = Math.sqrt(Math.pow(centerRow, 2) + Math.pow(centerCol, 2));
      const normalizedDistance = distanceFromCenter / maxDistance;
      
      const scale = 2 + (5 - 2) * (1 - normalizedDistance);
      const opacity = 1 - normalizedDistance * 0.9;
      const hue = isDark ? 4 : 220;
      const lightness = isDark ? (80 - normalizedDistance * 60) : (40 + normalizedDistance * 40);
      const shadowIntensity = Math.round(8 - normalizedDistance * 7);
      
      el.style.transform = `scale(${scale})`;
      el.style.opacity = opacity.toString();
      el.style.background = `hsl(${hue}, 70%, ${lightness}%)`;
      el.style.boxShadow = `0px 0px ${shadowIntensity}em 0px hsl(${hue}, 70%, ${lightness}%)`;
      el.style.zIndex = Math.round(rows * rows - normalizedDistance * rows * rows).toString();
    });

    const pulse = () => {
      particleEls.forEach((el, i) => {
        const row = Math.floor(i / rows);
        const col = i % rows;
        const centerRow = Math.floor(rows / 2);
        const centerCol = Math.floor(rows / 2);
        const distanceFromCenter = Math.sqrt(Math.pow(row - centerRow, 2) + Math.pow(col - centerCol, 2));
        const maxDistance = Math.sqrt(Math.pow(centerRow, 2) + Math.pow(centerCol, 2));
        const normalizedDistance = distanceFromCenter / maxDistance;
        const delay = 1650 + normalizedDistance * 90;
        
        setTimeout(() => {
          el.style.transform = `translate3d(${cursor.x}px, ${cursor.y}px, 0px) scale(5)`;
          el.style.opacity = '1';
          
          setTimeout(() => {
            const scale = 2 + (5 - 2) * (1 - normalizedDistance);
            const opacity = 1 - normalizedDistance * 0.9;
            el.style.transform = `translate3d(${cursor.x}px, ${cursor.y}px, 0px) scale(${scale})`;
            el.style.opacity = opacity.toString();
          }, 150);
        }, delay);
      });
    };

    const updateParticles = () => {
      particleEls.forEach((el, i) => {
        const row = Math.floor(i / rows);
        const col = i % rows;
        const centerRow = Math.floor(rows / 2);
        const centerCol = Math.floor(rows / 2);
        const distanceFromCenter = Math.sqrt(Math.pow(row - centerRow, 2) + Math.pow(col - centerCol, 2));
        const maxDistance = Math.sqrt(Math.pow(centerRow, 2) + Math.pow(centerCol, 2));
        const normalizedDistance = distanceFromCenter / maxDistance;
        const delay = normalizedDistance * 40;
        const duration = 120 + 750 - normalizedDistance * 630;
        
        setTimeout(() => {
          el.style.transition = `transform ${duration}ms ease-in-out`;
          const scale = 2 + (5 - 2) * (1 - normalizedDistance);
          el.style.transform = `translate3d(${cursor.x}px, ${cursor.y}px, 0px) scale(${scale})`;
        }, delay);
      });
    };

    const animate = () => {
      currentTime += 16.67; // ~60fps
      
      if (isAutoMovingRef.current) {
        const timeInSeconds = currentTime * 0.001;
        
        // X movement with complex sine waves
        const xBase = Math.sin(timeInSeconds * 0.0007) * viewport.w * 0.45;
        const xModifier = Math.sin(timeInSeconds * 0.5) * viewport.w * 0.45;
        cursor.x = xBase + xModifier;
        
        // Y movement with complex cosine waves  
        const yBase = Math.cos(timeInSeconds * 0.00012) * viewport.h * 0.45;
        const yModifier = Math.cos(timeInSeconds * 0.3) * viewport.h * 0.45;
        cursor.y = yBase + yModifier;
        
        // Trigger pulse every 3 seconds
        if (Math.floor(timeInSeconds) % 3 === 0 && Math.floor(timeInSeconds * 10) % 30 === 0) {
          pulse();
        }
      }
      
      updateParticles();
      animationRef.current = requestAnimationFrame(animate);
    };

    const followPointer = (e: MouseEvent | TouchEvent) => {
      const event = e.type === 'touchmove' ? (e as TouchEvent).touches[0] : e as MouseEvent;
      cursor.x = event.pageX - viewport.w;
      cursor.y = event.pageY - viewport.h;
      isAutoMovingRef.current = false;
      
      if (manualTimeoutRef.current) {
        clearTimeout(manualTimeoutRef.current);
      }
      manualTimeoutRef.current = setTimeout(() => {
        isAutoMovingRef.current = true;
      }, 1500);
    };

    document.addEventListener('mousemove', followPointer);
    document.addEventListener('touchmove', followPointer);
    
    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      document.removeEventListener('mousemove', followPointer);
      document.removeEventListener('touchmove', followPointer);
      if (manualTimeoutRef.current) {
        clearTimeout(manualTimeoutRef.current);
      }
    };
  }, [isDark]);

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      <div className="flex justify-center items-center absolute inset-0">
        <div
          ref={creatureRef}
          className="flex justify-center items-center flex-wrap"
          style={{
            fontSize: '0.2vh',
            width: '150em',
            height: '150em'
          }}
        />
      </div>
      
      <style jsx>{`
        .creature-particle {
          transform-style: preserve-3d;
          position: relative;
          width: 4em;
          height: 4em;
          margin: 3em;
          border-radius: 2em;
          will-change: transform;
          mix-blend-mode: plus-lighter;
          transition: transform 0.75s ease-in-out;
        }
      `}</style>
    </div>
  );
}