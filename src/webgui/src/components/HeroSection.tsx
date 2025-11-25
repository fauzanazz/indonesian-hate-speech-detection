"use client";

import { Button } from "@/components/ui/button";
import { Shield, Zap, Lock } from "lucide-react";

const HeroSection = () => {
  const scrollToTest = () => {
    const element = document.getElementById("test");
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <section className="min-h-screen flex items-center justify-center relative overflow-hidden pt-16">
      {/* Depth-based decorative layers */}
      <div
        className="absolute inset-0 opacity-10"
        style={{
          backgroundImage: 'radial-gradient(circle at 2px 2px, hsl(var(--gradient-ocean-blue)) 1px, transparent 0)',
          backgroundSize: '40px 40px',
          willChange: 'transform'
        }}
      />
      <div
        className="absolute top-1/4 left-1/4 w-96 h-96 rounded-full blur-3xl animate-pulse-glow"
        style={{
          background: 'radial-gradient(circle, hsla(var(--gradient-ocean-blue), 0.3), transparent)',
          willChange: 'transform, opacity'
        }}
      />
      <div
        className="absolute bottom-1/4 right-1/4 w-96 h-96 rounded-full blur-3xl animate-gradient-shift"
        style={{
          background: 'radial-gradient(circle, hsla(var(--gradient-light-cyan), 0.25), transparent)',
          willChange: 'transform, opacity'
        }}
      />

      <div className="container mx-auto px-4 md:px-6 relative z-10">
        <div className="max-w-4xl mx-auto text-center animate-fade-in-up">
          <div
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full mb-6 backdrop-blur-md"
            style={{
              background: 'hsla(var(--gradient-ocean-blue), 0.15)',
              border: '1px solid hsla(var(--gradient-light-cyan), 0.3)',
              boxShadow: '0 4px 24px hsla(var(--gradient-ocean-blue), 0.2)'
            }}
          >
            <Shield className="w-4 h-4" style={{ color: 'hsl(var(--gradient-sky-blue))' }} />
            <span className="text-sm font-medium" style={{ color: 'hsl(var(--gradient-sky-blue))' }}>Real-time Toxicity Detection</span>
          </div>

          <h1
            className="text-5xl md:text-7xl font-bold mb-6 leading-tight"
            style={{
              color: 'hsl(0 0% 100%)',
              textShadow: '0 2px 20px hsla(var(--gradient-ocean-blue), 0.5)'
            }}
          >
            Protect Your Community
            <span
              className="block"
              style={{
                color: 'hsl(var(--gradient-sky-blue))',
                textShadow: '0 2px 24px hsla(var(--gradient-light-cyan), 0.6)'
              }}
            >
              with AI-Powered Moderation
            </span>
          </h1>

          <p
            className="text-xl md:text-2xl mb-8 max-w-2xl mx-auto"
            style={{
              color: 'hsl(0 0% 95%)',
              textShadow: '0 1px 8px hsla(var(--gradient-deep-navy), 0.5)'
            }}
          >
            ToxiShield uses advanced machine learning to detect and filter toxic content in real-time, keeping your platform safe and welcoming.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-12">
            <Button
              size="lg"
              onClick={scrollToTest}
              className="font-semibold px-8 transition-all duration-300 hover:scale-105 backdrop-blur-sm"
              style={{
                background: 'linear-gradient(135deg, hsl(var(--gradient-ocean-blue)), hsl(var(--gradient-light-cyan)))',
                color: 'hsl(0 0% 100%)',
                boxShadow: '0 8px 32px hsla(var(--gradient-ocean-blue), 0.4)',
                willChange: 'transform'
              }}
            >
              Try it Now
            </Button>
            <Button
              size="lg"
              variant="outline"
              className="transition-all duration-300 hover:scale-105 backdrop-blur-md"
              style={{
                borderColor: 'hsla(var(--gradient-light-cyan), 0.4)',
                color: 'hsl(0 0% 100%)',
                background: 'hsla(var(--gradient-deep-navy), 0.3)',
                willChange: 'transform'
              }}
            >
              View Documentation
            </Button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-3xl mx-auto">
            <div
              className="flex flex-col items-center gap-2 p-4 backdrop-blur-md rounded-lg transition-all duration-300 hover:scale-105"
              style={{
                background: 'hsla(var(--gradient-ocean-blue), 0.15)',
                border: '1px solid hsla(var(--gradient-light-cyan), 0.2)',
                boxShadow: '0 4px 16px hsla(var(--gradient-deep-navy), 0.3)',
                willChange: 'transform'
              }}
            >
              <Zap className="w-8 h-8" style={{ color: 'hsl(var(--gradient-sky-blue))' }} />
              <h3 className="font-semibold" style={{ color: 'hsl(0 0% 100%)' }}>Real-time Analysis</h3>
              <p className="text-sm" style={{ color: 'hsl(0 0% 85%)' }}>Instant toxicity detection</p>
            </div>
            <div
              className="flex flex-col items-center gap-2 p-4 backdrop-blur-md rounded-lg transition-all duration-300 hover:scale-105"
              style={{
                background: 'hsla(var(--gradient-ocean-blue), 0.15)',
                border: '1px solid hsla(var(--gradient-light-cyan), 0.2)',
                boxShadow: '0 4px 16px hsla(var(--gradient-deep-navy), 0.3)',
                willChange: 'transform'
              }}
            >
              <Shield className="w-8 h-8" style={{ color: 'hsl(var(--gradient-sky-blue))' }} />
              <h3 className="font-semibold" style={{ color: 'hsl(0 0% 100%)' }}>Multi-language Support</h3>
              <p className="text-sm" style={{ color: 'hsl(0 0% 85%)' }}>Works across 100+ languages</p>
            </div>
            <div
              className="flex flex-col items-center gap-2 p-4 backdrop-blur-md rounded-lg transition-all duration-300 hover:scale-105"
              style={{
                background: 'hsla(var(--gradient-ocean-blue), 0.15)',
                border: '1px solid hsla(var(--gradient-light-cyan), 0.2)',
                boxShadow: '0 4px 16px hsla(var(--gradient-deep-navy), 0.3)',
                willChange: 'transform'
              }}
            >
              <Lock className="w-8 h-8" style={{ color: 'hsl(var(--gradient-sky-blue))' }} />
              <h3 className="font-semibold" style={{ color: 'hsl(0 0% 100%)' }}>Privacy First</h3>
              <p className="text-sm" style={{ color: 'hsl(0 0% 85%)' }}>Your data stays secure</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
