"use client";

import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { MessageCircleHeart, SearchIcon, ShieldAlert } from "lucide-react";

const features = [
  {
    id: 1,
    title: "Counter-Speech Generation",
    icon: MessageCircleHeart, 
    description: "Mengubah ujaran kebencian menjadi respon yang lebih positif, empatik, dan edukatif secara otomatis.",
    details: "Fitur ini mendeteksi konten berbahaya atau diskriminatif lalu menghasilkan counter-speech — yaitu pernyataan penangkal yang mendorong empati, klarifikasi fakta, dan penghentian eskalasi konflik. Cocok untuk chatbot publik, moderasi komentar, sistem edukasi, maupun platform komunitas yang ingin menjaga percakapan tetap sehat.",
    stats: { 
      accuracy: "92.5%", 
      speed: "15ms per request" 
    }
  },  
  {
    id: 2,
    title: "Toxic Detection",
    icon: ShieldAlert,
    description: "Model klasifikasi untuk mendeteksi ujaran berbahaya seperti hate speech, penghinaan, atau konten agresif.",
    details: "Fitur ini menganalisis teks dan memberikan label apakah suatu kalimat mengandung ujaran toxic beserta tingkat keyakinannya. Mendukung berbagai kategori seperti hate speech, harassment, abusive, dan offensive language. Dipakai untuk moderasi komentar real-time, filter percakapan, dan automoderation platform besar.",
    stats: { 
      accuracy: "95.8%", 
      speed: "8ms per request" 
    }
  },
  {
    id: 3,
    title: "Search and Filter",
    icon: SearchIcon,
    description: "Mencari dan memfilter konten berbahaya berdasarkan kata kunci atau kategori.",
    details: "Fitur ini membantu pengguna mencari konten berbahaya berdasarkan kata kunci atau kategori. Cocok untuk moderasi komentar real-time, filter percakapan, dan automoderation platform besar.",
    stats: { 
      accuracy: "95.8%", 
       speed: "8ms per request" 
    }
  }
];

const FeaturesSection = () => {
  const [activeFeature, setActiveFeature] = useState(0);

  return (
    <section id="features" className="py-20 relative backdrop-blur-sm">
      <div className="container mx-auto px-4 md:px-6">
        <div className="text-center mb-12 animate-fade-in">
          <h2
            className="text-4xl md:text-5xl font-bold mb-4"
            style={{
              color: 'hsl(0 0% 100%)',
              textShadow: '0 2px 16px hsla(var(--gradient-ocean-blue), 0.4)'
            }}
          >
            Powerful Features
          </h2>
          <p
            className="text-xl max-w-2xl mx-auto"
            style={{ color: 'hsl(0 0% 90%)' }}
          >
            Comprehensive toxicity detection for every use case
          </p>
        </div>

        <div className="flex flex-wrap justify-center gap-3 mb-8">
          {features.map((feature, index) => (
            <Button
              key={feature.id}
              variant={activeFeature === index ? "default" : "outline"}
              onClick={() => setActiveFeature(index)}
              className={`transition-all duration-300 ${
                activeFeature === index 
                  ? "bg-primary text-primary-foreground shadow-lg border-2 border-primary hover:bg-primary/90" 
                  : "bg-card text-foreground border-2 border-border hover:bg-card/80 hover:border-primary/50"
              }`}
            >
              <feature.icon className="w-4 h-4 mr-2" />
              Feature {feature.id}
            </Button>
          ))}
        </div>

        <Card
          className="p-8 md:p-12 animate-scale-in backdrop-blur-lg"
          style={{
            background: 'hsla(var(--gradient-deep-indigo), 0.4)',
            border: '1px solid hsla(var(--gradient-light-cyan), 0.2)',
            boxShadow: '0 8px 32px hsla(var(--gradient-deep-navy), 0.4)'
          }}
        >
          <div className="grid md:grid-cols-2 gap-8 items-center">
            <div>
              <div className="flex items-center gap-3 mb-4">
                {(() => {
                  const IconComponent = features[activeFeature].icon;
                  return (
                    <div
                      className="p-3 rounded-lg"
                      style={{
                        background: 'hsla(var(--gradient-ocean-blue), 0.3)',
                        border: '1px solid hsla(var(--gradient-light-cyan), 0.3)'
                      }}
                    >
                      <IconComponent
                        className="w-8 h-8"
                        style={{ color: 'hsl(var(--gradient-sky-blue))' }}
                      />
                    </div>
                  );
                })()}
                <h3
                  className="text-3xl font-bold"
                  style={{ color: 'hsl(0 0% 100%)' }}
                >
                  {features[activeFeature].title}
                </h3>
              </div>
              <p
                className="text-lg mb-6"
                style={{ color: 'hsl(0 0% 90%)' }}
              >
                {features[activeFeature].description}
              </p>
              <p
                className="mb-6"
                style={{ color: 'hsl(0 0% 85%)' }}
              >
                {features[activeFeature].details}
              </p>
              <div className="flex gap-6">
                <div>
                  <div
                    className="text-2xl font-bold"
                    style={{ color: 'hsl(var(--gradient-light-cyan))' }}
                  >
                    {features[activeFeature].stats.accuracy}
                  </div>
                  <div
                    className="text-sm"
                    style={{ color: 'hsl(0 0% 80%)' }}
                  >
                    Accuracy
                  </div>
                </div>
                <div>
                  <div
                    className="text-2xl font-bold"
                    style={{ color: 'hsl(var(--gradient-sky-blue))' }}
                  >
                    {features[activeFeature].stats.speed}
                  </div>
                  <div
                    className="text-sm"
                    style={{ color: 'hsl(0 0% 80%)' }}
                  >
                    Response Time
                  </div>
                </div>
              </div>
            </div>
            <div
              className="rounded-lg p-8 min-h-[300px] flex items-center justify-center backdrop-blur-sm"
              style={{
                background: 'hsla(var(--gradient-ocean-blue), 0.2)',
                border: '1px solid hsla(var(--gradient-light-cyan), 0.2)'
              }}
            >
              <div className="text-center">
                {(() => {
                  const IconComponent = features[activeFeature].icon;
                  return <IconComponent
                    className="w-24 h-24 mx-auto mb-4 opacity-30"
                    style={{ color: 'hsl(var(--gradient-sky-blue))' }}
                  />;
                })()}
                <p style={{ color: 'hsl(0 0% 80%)' }}>Feature visualization</p>
              </div>
            </div>
          </div>
        </Card>
      </div>
    </section>
  );
};

export default FeaturesSection;
