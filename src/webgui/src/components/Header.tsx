"use client";

import { useState, useEffect } from "react";
import { usePathname } from "next/navigation";
import Link from "next/link";
import { Menu, X, Moon, Sun } from "lucide-react";
import { useTheme } from "@/hooks/useTheme";

const Header = () => {
  const { theme, toggleTheme } = useTheme();
  const pathname = usePathname();
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const isHomePage = pathname === "/";
  const isDashboardPage = pathname === "/dataset-eda";

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
          <Link href="/" className="text-2xl font-bold text-primary">
            ToxiShield
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center gap-8">
            {isHomePage ? (
              <>
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
                <Link href="/dataset-eda" className="text-foreground/80 hover:text-foreground transition-colors">
                  Dashboard
                </Link>
              </>
            ) : isDashboardPage ? (
              <Link href="/" className="text-foreground/80 hover:text-foreground transition-colors">
                Home
              </Link>
            ) : null}
            <button
              onClick={toggleTheme}
              className="text-foreground/80 hover:text-foreground transition-colors"
              aria-label="Toggle theme"
            >
              {theme === "dark" ? <Sun size={20} /> : <Moon size={20} />}
            </button>
          </nav>

          {/* Mobile Menu and Theme Toggle */}
          <div className="md:hidden flex items-center gap-4">
            <button
              onClick={toggleTheme}
              className="text-foreground"
              aria-label="Toggle theme"
            >
              {theme === "dark" ? <Sun size={20} /> : <Moon size={20} />}
            </button>
            <button
              className="text-foreground"
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              aria-label="Toggle menu"
            >
              {isMobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {isMobileMenuOpen && (
          <nav className="md:hidden py-4 border-t border-border">
            <div className="flex flex-col gap-4">
              {isHomePage ? (
                <>
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
                  <Link href="/dataset-eda" className="text-foreground/80 hover:text-foreground transition-colors text-left">
                    Dashboard
                  </Link>
                </>
              ) : isDashboardPage ? (
                <Link href="/" className="text-foreground/80 hover:text-foreground transition-colors text-left">
                  Home
                </Link>
              ) : null}
            </div>
          </nav>
        )}
      </div>
    </header>
  );
};

export default Header;
