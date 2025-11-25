"use client";

import { useEffect, useRef, useState } from "react";

export const AnimatedGradientBackground = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | undefined>(undefined);
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(() => {
    if (typeof window !== "undefined") {
      return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    }
    return false;
  });

  useEffect(() => {
    const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");

    const handleChange = (e: MediaQueryListEvent | MediaQueryList) => {
      setPrefersReducedMotion(e.matches);
    };

    mediaQuery.addEventListener("change", handleChange);
    return () => mediaQuery.removeEventListener("change", handleChange);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d", { alpha: false });
    if (!ctx) return;

    const resizeCanvas = () => {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = window.innerWidth * dpr;
      canvas.height = window.innerHeight * dpr;
      canvas.style.width = `${window.innerWidth}px`;
      canvas.style.height = `${window.innerHeight}px`;
      ctx.scale(dpr, dpr);
    };

    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    let time = 0;
    let scrollY = 0;

    const handleScroll = () => {
      scrollY = window.scrollY;
    };

    window.addEventListener("scroll", handleScroll, { passive: true });

    const animate = () => {
      if (prefersReducedMotion) {
        time = 0;
      } else {
        time += 0.003;
      }

      const width = window.innerWidth;
      const height = window.innerHeight;
      const scrollDepth = Math.min(scrollY / (document.body.scrollHeight - height), 1);

      // Create depth-based gradient from deep navy to light cyan
      const gradient = ctx.createLinearGradient(0, 0, 0, height);

      // Deep navy/indigo at the top (deeper sections)
      const deepHue = 220 + Math.sin(time) * 10;
      const deepSat = 70 + Math.sin(time * 0.7) * 10;
      const deepLight = 15 + scrollDepth * 5;

      // Mid-level blue
      const midHue = 210 + Math.cos(time * 1.2) * 8;
      const midSat = 60 + Math.sin(time * 0.9) * 10;
      const midLight = 30 + scrollDepth * 10;

      // Light cyan/sky blue at the bottom (shallower areas)
      const lightHue = 195 + Math.sin(time * 0.8) * 5;
      const lightSat = 50 + Math.cos(time * 1.1) * 8;
      const lightLight = 50 + scrollDepth * 15;

      gradient.addColorStop(0, `hsl(${deepHue}, ${deepSat}%, ${deepLight}%)`);
      gradient.addColorStop(0.5, `hsl(${midHue}, ${midSat}%, ${midLight}%)`);
      gradient.addColorStop(1, `hsl(${lightHue}, ${lightSat}%, ${lightLight}%)`);

      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, width, height);

      // Add animated depth layers with radial gradients
      const layers = [
        {
          x: width * (0.2 + Math.sin(time * 0.5) * 0.1),
          y: height * (0.3 + Math.cos(time * 0.3) * 0.1),
          radius: Math.min(width, height) * (0.4 + Math.sin(time * 0.6) * 0.1),
          color: `hsla(210, 65%, ${25 + scrollDepth * 8}%, 0.15)`,
        },
        {
          x: width * (0.7 + Math.cos(time * 0.4) * 0.1),
          y: height * (0.6 + Math.sin(time * 0.5) * 0.1),
          radius: Math.min(width, height) * (0.35 + Math.cos(time * 0.7) * 0.08),
          color: `hsla(200, 60%, ${35 + scrollDepth * 10}%, 0.12)`,
        },
        {
          x: width * (0.5 + Math.sin(time * 0.6) * 0.08),
          y: height * (0.8 + Math.cos(time * 0.4) * 0.08),
          radius: Math.min(width, height) * (0.3 + Math.sin(time * 0.8) * 0.07),
          color: `hsla(190, 55%, ${45 + scrollDepth * 12}%, 0.1)`,
        },
      ];

      layers.forEach((layer) => {
        const radialGradient = ctx.createRadialGradient(
          layer.x,
          layer.y,
          0,
          layer.x,
          layer.y,
          layer.radius
        );
        radialGradient.addColorStop(0, layer.color);
        radialGradient.addColorStop(1, "transparent");
        ctx.fillStyle = radialGradient;
        ctx.fillRect(0, 0, width, height);
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener("resize", resizeCanvas);
      window.removeEventListener("scroll", handleScroll);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [prefersReducedMotion]);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 -z-10"
      style={{
        willChange: "transform",
        transform: "translateZ(0)",
      }}
      aria-hidden="true"
    />
  );
};