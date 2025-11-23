"use client";

import { useState, useEffect } from "react";
import { Menu, X } from "lucide-react";

const Header = () => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
      setIsMobileMenuOpen(false);
    }
  };

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        isScrolled ? "bg-background/80 backdrop-blur-lg border-b border-border shadow-sm" : "bg-transparent"
      }`}
    >
      <div className="container mx-auto px-4 md:px-6">
        <div className="flex items-center justify-between h-16">
          <div className="text-2xl font-bold text-primary">ToxiShield</div>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center gap-8">
            <button onClick={() => scrollToSection("features")} className="text-foreground/80 hover:text-foreground transition-colors">
              Feature
            </button>
            <button onClick={() => scrollToSection("test")} className="text-foreground/80 hover:text-foreground transition-colors">
              Test
            </button>
            <button onClick={() => scrollToSection("model")} className="text-foreground/80 hover:text-foreground transition-colors">
              Model
            </button>
            <button onClick={() => scrollToSection("about")} className="text-foreground/80 hover:text-foreground transition-colors">
              About Us
            </button>
          </nav>

          {/* Mobile Menu Button */}
          <button
            className="md:hidden text-foreground"
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            aria-label="Toggle menu"
          >
            {isMobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>

        {/* Mobile Navigation */}
        {isMobileMenuOpen && (
          <nav className="md:hidden py-4 border-t border-border">
            <div className="flex flex-col gap-4">
              <button onClick={() => scrollToSection("features")} className="text-foreground/80 hover:text-foreground transition-colors text-left">
                Feature
              </button>
              <button onClick={() => scrollToSection("test")} className="text-foreground/80 hover:text-foreground transition-colors text-left">
                Test
              </button>
              <button onClick={() => scrollToSection("model")} className="text-foreground/80 hover:text-foreground transition-colors text-left">
                Model
              </button>
              <button onClick={() => scrollToSection("about")} className="text-foreground/80 hover:text-foreground transition-colors text-left">
                About Us
              </button>
            </div>
          </nav>
        )}
      </div>
    </header>
  );
};

export default Header;
