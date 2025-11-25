"use client";

import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

const models = [
  {
    id: 1,
    name: "TF-IDF + Logistic Regression",
    description: "Tier 1 basic filter optimized for high-volume moderation.",
    details:
      "Sparse lexical features with calibrated logistic regression. Best when you need instant responses for chat widgets, inbox triage, or SMS pipelines and can tolerate a conservative first-pass filter.",
    specs: {
      speed: "~1–5 ms latency",
      accuracy: "F1 ≈ 0.90 (val)",
      languages: "Indonesian (formal & slang)",
      categories: "Binary toxicity flag"
    }
  },
  {
    id: 2,
    name: "BiLSTM Contextual",
    description: "Tier 2 sequential model that captures nuance and phrasing.",
    details:
      "Character + word embeddings feed a bidirectional LSTM with attention, enabling context-aware predictions. Ideal for community platforms or review systems that need balance between throughput and nuance.",
    specs: {
      speed: "~10–50 ms latency",
      accuracy: "F1 ≈ 0.93 (val)",
      languages: "Indonesian + colloquial variants",
      categories: "Binary + subtle toxicity tags"
    }
  },
  {
    id: 3,
    name: "IndoBERT Sociolinguistic",
    description: "Tier 3 transformer for maximum accuracy and fairness audits.",
    details:
      "Fine-tuned indobenchmark/indobert-base-p1 with calibration and fairness monitoring. Recommended for policy teams, trust-and-safety desks, and any workflow where recall on edge cases matters more than raw speed.",
    specs: {
      speed: "~50–200 ms latency",
      accuracy: "F1 ≈ 0.95 (val)",
      languages: "Indonesian + regional slang",
      categories: "Binary + sociolinguistic cues"
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
