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
    <section className="min-h-screen flex items-center justify-center bg-gradient-hero relative overflow-hidden pt-16">
      {/* Background decoration */}
      <div className="absolute inset-0 bg-grid-pattern opacity-5"></div>
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-accent/20 rounded-full blur-3xl"></div>
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-primary/20 rounded-full blur-3xl"></div>

      <div className="container mx-auto px-4 md:px-6 relative z-10">
        <div className="max-w-4xl mx-auto text-center animate-fade-in-up">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-accent/10 border border-accent/20 rounded-full mb-6">
            <Shield className="w-4 h-4 text-accent" />
            <span className="text-sm text-accent font-medium">Real-time Toxicity Detection</span>
          </div>

          <h1 className="text-5xl md:text-7xl font-bold mb-6 text-primary-foreground leading-tight">
            Protect Your Community
            <span className="block text-accent">with AI-Powered Moderation</span>
          </h1>

          <p className="text-xl md:text-2xl text-primary-foreground/80 mb-8 max-w-2xl mx-auto">
            ToxiShield uses advanced machine learning to detect and filter toxic content in real-time, keeping your platform safe and welcoming.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-12">
            <Button 
              size="lg" 
              onClick={scrollToTest}
              className="bg-accent hover:bg-accent/90 text-accent-foreground font-semibold px-8 shadow-glow hover:shadow-elevated transition-all duration-300 hover:scale-105"
            >
              Try it Now
            </Button>
            <Button 
              size="lg" 
              variant="outline"
              className="border-primary-foreground/20 text-primary-foreground hover:bg-primary-foreground/10"
            >
              View Documentation
            </Button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-3xl mx-auto">
            <div className="flex flex-col items-center gap-2 p-4 bg-card/5 backdrop-blur-sm border border-border/10 rounded-lg">
              <Zap className="w-8 h-8 text-accent" />
              <h3 className="font-semibold text-primary-foreground">Real-time Analysis</h3>
              <p className="text-sm text-primary-foreground/70">Instant toxicity detection</p>
            </div>
            <div className="flex flex-col items-center gap-2 p-4 bg-card/5 backdrop-blur-sm border border-border/10 rounded-lg">
              <Shield className="w-8 h-8 text-accent" />
              <h3 className="font-semibold text-primary-foreground">Multi-language Support</h3>
              <p className="text-sm text-primary-foreground/70">Works across 100+ languages</p>
            </div>
            <div className="flex flex-col items-center gap-2 p-4 bg-card/5 backdrop-blur-sm border border-border/10 rounded-lg">
              <Lock className="w-8 h-8 text-accent" />
              <h3 className="font-semibold text-primary-foreground">Privacy First</h3>
              <p className="text-sm text-primary-foreground/70">Your data stays secure</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
