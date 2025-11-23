"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { AlertTriangle, CheckCircle, Loader2 } from "lucide-react";
import { toxicityService, ToxicityResponse } from "@/services/api";

const ToxicityTest = () => {
  const [text, setText] = useState("");
  const [tier, setTier] = useState<"basic" | "contextual" | "sociolinguistic" | "ensemble">("ensemble");
  const [result, setResult] = useState<ToxicityResponse | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const analyzeText = async () => {
    if (!text.trim()) return;
    
    setIsAnalyzing(true);
    setError(null);
    
    try {
      let response: ToxicityResponse;
      
      switch (tier) {
        case "basic":
          response = await toxicityService.detectBasic(text, true);
          break;
        case "contextual":
          response = await toxicityService.detectContextual(text, true);
          break;
        case "sociolinguistic":
          response = await toxicityService.detectSociolinguistic(text, true);
          break;
        case "ensemble":
          response = await toxicityService.detectEnsemble(text, true);
          break;
      }
      
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to analyze text");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="space-y-6">
      <Card className="p-6 bg-card border-border shadow-elevated">
        <div className="space-y-4">
          <div>
            <label className="text-sm font-medium mb-2 block">Model Tier</label>
            <Select value={tier} onValueChange={(value: typeof tier) => setTier(value)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="basic">Basic (TF-IDF)</SelectItem>
                <SelectItem value="contextual">Contextual (BiLSTM)</SelectItem>
                <SelectItem value="sociolinguistic">Sociolinguistic (IndoBERT)</SelectItem>
                <SelectItem value="ensemble">Ensemble (All Models)</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <Textarea
            placeholder="Enter Indonesian text to analyze for toxicity..."
            value={text}
            onChange={(e) => setText(e.target.value)}
            className="min-h-[150px] text-base resize-none"
          />

          <div className="flex justify-end">
            <Button 
              onClick={analyzeText}
              disabled={!text.trim() || isAnalyzing}
              className="bg-primary hover:bg-primary/90"
            >
              {isAnalyzing && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
              {isAnalyzing ? "Analyzing..." : "Analyze Toxicity"}
            </Button>
          </div>
        </div>
      </Card>

      <Card className="p-8 bg-card border-border shadow-elevated min-h-[200px]">
        {error ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center text-destructive">
              <AlertTriangle className="w-12 h-12 mx-auto mb-4" />
              <p className="text-lg font-semibold mb-2">Error</p>
              <p className="text-sm">{error}</p>
            </div>
          </div>
        ) : !result ? (
          <div className="flex items-center justify-center h-full text-muted-foreground">
            <p className="text-lg">Results will appear here</p>
          </div>
        ) : (
          <div className="animate-fade-in space-y-6">
            <div className="flex items-start gap-4">
              {result.is_toxic ? (
                <AlertTriangle className="w-8 h-8 text-destructive flex-shrink-0" />
              ) : (
                <CheckCircle className="w-8 h-8 text-green-500 flex-shrink-0" />
              )}
              <div className="flex-1">
                <h3 className="text-2xl font-bold mb-2">
                  {result.is_toxic ? "Toxic Content Detected" : "Content is Safe"}
                </h3>
                <p className="text-muted-foreground">
                  {result.is_toxic 
                    ? "This text contains potentially harmful or offensive language."
                    : "This text appears to be respectful and appropriate."}
                </p>
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-4">
              <div className="p-4 bg-muted/50 rounded-lg">
                <div className="text-sm text-muted-foreground mb-1">Toxicity Score</div>
                <div className="text-3xl font-bold text-foreground">
                  {(result.toxicity_score * 100).toFixed(1)}%
                </div>
              </div>

              <div className="p-4 bg-muted/50 rounded-lg">
                <div className="text-sm text-muted-foreground mb-1">Confidence</div>
                <div className="text-3xl font-bold text-foreground capitalize">
                  {result.confidence}
                </div>
              </div>

              <div className="p-4 bg-muted/50 rounded-lg">
                <div className="text-sm text-muted-foreground mb-1">Model</div>
                <div className="text-lg font-semibold text-foreground">
                  {result.model_name}
                </div>
              </div>

              <div className="p-4 bg-muted/50 rounded-lg">
                <div className="text-sm text-muted-foreground mb-1">Latency</div>
                <div className="text-lg font-semibold text-foreground">
                  {result.latency_ms.toFixed(2)} ms
                </div>
              </div>
            </div>

            {result.explanation && (
              <div className="p-4 bg-muted/30 rounded-lg">
                <div className="text-sm font-medium mb-2">Model Explanation</div>
                <pre className="text-xs overflow-auto max-h-48 bg-background/50 p-3 rounded">
                  {JSON.stringify(result.explanation, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}
      </Card>
    </div>
  );
};

export default ToxicityTest;