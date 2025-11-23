"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card } from "@/components/ui/card";
import { Upload, AlertTriangle, CheckCircle } from "lucide-react";

const TestSection = () => {
  const [text, setText] = useState("");
  const [result, setResult] = useState<{ toxic: boolean; score: number; categories?: string[] } | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const analyzeText = () => {
    if (!text.trim()) return;
    
    setIsAnalyzing(true);
    
    // Simulate API call
    setTimeout(() => {
      // Simple toxicity detection simulation
      const toxicKeywords = ["hate", "stupid", "idiot", "kill", "die", "worst"];
      const lowerText = text.toLowerCase();
      const isToxic = toxicKeywords.some(keyword => lowerText.includes(keyword));
      
      setResult({
        toxic: isToxic,
        score: isToxic ? Math.random() * 0.5 + 0.5 : Math.random() * 0.3,
        categories: isToxic ? ["offensive_language", "insult"] : []
      });
      setIsAnalyzing(false);
    }, 1000);
  };

  return (
    <section id="test" className="py-20 bg-muted/30">
      <div className="container mx-auto px-4 md:px-6">
        <div className="text-center mb-12 animate-fade-in">
          <h2 className="text-4xl md:text-5xl font-bold mb-4 text-foreground">Test</h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Try our toxicity detection in action
          </p>
        </div>

        <div className="max-w-3xl mx-auto">
          <Card className="p-6 mb-6 bg-card border-border shadow-elevated">
            <Textarea
              placeholder="Enter your text here..."
              value={text}
              onChange={(e) => setText(e.target.value)}
              className="min-h-[150px] mb-4 text-base resize-none"
            />
            <div className="flex gap-3 justify-end">
              <Button variant="outline" size="sm">
                <Upload className="w-4 h-4 mr-2" />
                Upload
              </Button>
              <Button 
                onClick={analyzeText}
                disabled={!text.trim() || isAnalyzing}
                className="bg-primary hover:bg-primary/90"
              >
                {isAnalyzing ? "Analyzing..." : "Analyze"}
              </Button>
            </div>
          </Card>

          <Card className="p-8 bg-card border-border shadow-elevated min-h-[200px]">
            {!result ? (
              <div className="flex items-center justify-center h-full text-muted-foreground">
                <p className="text-lg">&lt;Result&gt;</p>
              </div>
            ) : (
              <div className="animate-fade-in">
                <div className="flex items-start gap-4 mb-6">
                  {result.toxic ? (
                    <AlertTriangle className="w-8 h-8 text-destructive flex-shrink-0" />
                  ) : (
                    <CheckCircle className="w-8 h-8 text-green-500 flex-shrink-0" />
                  )}
                  <div className="flex-1">
                    <h3 className="text-2xl font-bold mb-2">
                      {result.toxic ? "Toxic Content Detected" : "Content is Safe"}
                    </h3>
                    <p className="text-muted-foreground">
                      {result.toxic 
                        ? "This text contains potentially harmful or offensive language."
                        : "This text appears to be respectful and appropriate."}
                    </p>
                  </div>
                </div>

                <div className="grid md:grid-cols-2 gap-4">
                  <div className="p-4 bg-muted/50 rounded-lg">
                    <div className="text-sm text-muted-foreground mb-1">Toxicity Score</div>
                    <div className="text-3xl font-bold text-foreground">
                      {(result.score * 100).toFixed(1)}%
                    </div>
                  </div>
                  {result.categories && result.categories.length > 0 && (
                    <div className="p-4 bg-muted/50 rounded-lg">
                      <div className="text-sm text-muted-foreground mb-2">Categories</div>
                      <div className="flex flex-wrap gap-2">
                        {result.categories.map((cat, idx) => (
                          <span key={idx} className="px-3 py-1 bg-destructive/10 text-destructive text-sm rounded-full">
                            {cat.replace(/_/g, " ")}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </Card>
        </div>
      </div>
    </section>
  );
};

export default TestSection;
