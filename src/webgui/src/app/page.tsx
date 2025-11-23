import Header from "@/components/Header";
import HeroSection from "@/components/HeroSection";
import FeaturesSection from "@/components/FeaturesSection";
import TestSection from "@/components/TestSection";
import ModelSection from "@/components/ModelSection";
import AboutSection from "@/components/AboutSection";

const Index = () => {
  return (
    <div className="min-h-screen">
      <Header />
      <main>
        <HeroSection />
        <FeaturesSection />
        <TestSection />
        <ModelSection />
        <AboutSection />
      </main>
      <footer className="bg-card border-t border-border py-8">
        <div className="container mx-auto px-4 md:px-6 text-center text-muted-foreground">
          <p>&copy; 2024 ToxiShield. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
