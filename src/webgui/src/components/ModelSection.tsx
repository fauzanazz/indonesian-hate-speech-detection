"use client";

import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

const models = [
  {
    id: 1,
    name: "ToxiShield Lite",
    description: "Fast and efficient model for basic toxicity detection",
    details: "Our lightweight model is perfect for high-volume applications where speed is critical. It provides reliable toxicity detection with minimal latency, ideal for real-time chat applications and live comment sections.",
    specs: {
      speed: "< 30ms",
      accuracy: "94%",
      languages: "50+",
      categories: "5 core types"
    }
  },
  {
    id: 2,
    name: "ToxiShield Pro",
    description: "Balanced performance and accuracy for most use cases",
    details: "The recommended model for most applications, offering excellent balance between speed and accuracy. Detects nuanced toxicity patterns and provides detailed category breakdowns. Perfect for forums, reviews, and social platforms.",
    specs: {
      speed: "< 75ms",
      accuracy: "97%",
      languages: "100+",
      categories: "12 detailed types"
    }
  },
  {
    id: 3,
    name: "ToxiShield Max",
    description: "Maximum accuracy for critical content moderation",
    details: "Our most sophisticated model with state-of-the-art accuracy. Uses advanced transformer architecture to understand context, sarcasm, and subtle forms of toxicity. Best for enterprise applications requiring the highest moderation standards.",
    specs: {
      speed: "< 150ms",
      accuracy: "99%",
      languages: "120+",
      categories: "20+ granular types"
    }
  }
];

const ModelSection = () => {
  const [selectedModel, setSelectedModel] = useState(0);

  return (
    <section id="model" className="py-20 bg-background">
      <div className="container mx-auto px-4 md:px-6">
        <div className="text-center mb-12 animate-fade-in">
          <h2 className="text-4xl md:text-5xl font-bold mb-4 text-foreground">Model</h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Choose the right model for your needs
          </p>
        </div>

        <div className="grid md:grid-cols-4 gap-6">
          {/* Sidebar */}
          <div className="md:col-span-1">
            <Card className="p-4 bg-card border-border">
              <div className="space-y-2">
                {models.map((model, index) => (
                  <Button
                    key={model.id}
                    variant={selectedModel === index ? "default" : "ghost"}
                    className={`w-full justify-start transition-all duration-300 ${
                      selectedModel === index ? "bg-primary text-primary-foreground" : ""
                    }`}
                    onClick={() => setSelectedModel(index)}
                  >
                    Model {model.id}
                  </Button>
                ))}
              </div>
            </Card>
          </div>

          {/* Content */}
          <div className="md:col-span-3">
            <Card className="p-8 bg-card border-border shadow-elevated min-h-[400px] animate-scale-in">
              <h3 className="text-3xl font-bold mb-3 text-foreground">{models[selectedModel].name}</h3>
              <p className="text-lg text-muted-foreground mb-6">{models[selectedModel].description}</p>
              <p className="text-foreground/80 mb-8">{models[selectedModel].details}</p>

              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 bg-muted/50 rounded-lg">
                  <div className="text-sm text-muted-foreground mb-1">Response Time</div>
                  <div className="text-2xl font-bold text-primary">{models[selectedModel].specs.speed}</div>
                </div>
                <div className="p-4 bg-muted/50 rounded-lg">
                  <div className="text-sm text-muted-foreground mb-1">Accuracy</div>
                  <div className="text-2xl font-bold text-accent">{models[selectedModel].specs.accuracy}</div>
                </div>
                <div className="p-4 bg-muted/50 rounded-lg">
                  <div className="text-sm text-muted-foreground mb-1">Languages</div>
                  <div className="text-2xl font-bold text-primary">{models[selectedModel].specs.languages}</div>
                </div>
                <div className="p-4 bg-muted/50 rounded-lg">
                  <div className="text-sm text-muted-foreground mb-1">Categories</div>
                  <div className="text-2xl font-bold text-accent">{models[selectedModel].specs.categories}</div>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ModelSection;
