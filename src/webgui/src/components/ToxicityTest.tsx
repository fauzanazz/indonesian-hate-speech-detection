"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { AlertTriangle, CheckCircle, Loader2, MessageSquare } from "lucide-react";
import { toxicityService, ToxicityResponse, counterSpeechService, CounterSpeechResponse } from "@/services/api";

const ToxicityTest = () => {
  const [text, setText] = useState("");
  const [tier, setTier] = useState<"basic" | "contextual" | "sociolinguistic" | "ensemble">("ensemble");
  const [result, setResult] = useState<ToxicityResponse | null>(null);
  const [counterSpeech, setCounterSpeech] = useState<CounterSpeechResponse | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isGeneratingCounter, setIsGeneratingCounter] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [counterSpeechError, setCounterSpeechError] = useState<string | null>(null);

  const analyzeText = async () => {
    if (!text.trim()) return;
    
    setIsAnalyzing(true);
    setError(null);
    setCounterSpeech(null);
    setCounterSpeechError(null);
    
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
      
      // If toxic, automatically generate counter speech
      if (response.is_toxic) {
        await generateCounterSpeech(text);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to analyze text");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const generateCounterSpeech = async (textToCounter: string) => {
    setIsGeneratingCounter(true);
    setCounterSpeechError(null);
    
    try {
      const response = await counterSpeechService.generate(textToCounter, {
        numBeams: 4,
        maxLength: 128,
      });
      setCounterSpeech(response);
    } catch (err) {
      setCounterSpeechError(err instanceof Error ? err.message : "Failed to generate counter speech");
    } finally {
      setIsGeneratingCounter(false);
    }
  };

  return (
    <div className="space-y-6">
      <Card className="p-6 bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-700 shadow-lg">
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

      <Card className="p-8 bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-700 shadow-lg min-h-[200px]">
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
              <div className="p-4 bg-slate-100 dark:bg-slate-800 rounded-lg">
                <div className="text-sm text-muted-foreground mb-1">Toxicity Score</div>
                <div className="text-3xl font-bold text-foreground">
                  {(result.toxicity_score * 100).toFixed(1)}%
                </div>
              </div>

              <div className="p-4 bg-slate-100 dark:bg-slate-800 rounded-lg">
                <div className="text-sm text-muted-foreground mb-1">Confidence</div>
                <div className="text-3xl font-bold text-foreground capitalize">
                  {result.confidence}
                </div>
              </div>

              <div className="p-4 bg-slate-100 dark:bg-slate-800 rounded-lg">
                <div className="text-sm text-muted-foreground mb-1">Model</div>
                <div className="text-lg font-semibold text-foreground">
                  {result.model_name}
                </div>
              </div>

              <div className="p-4 bg-slate-100 dark:bg-slate-800 rounded-lg">
                <div className="text-sm text-muted-foreground mb-1">Latency</div>
                <div className="text-lg font-semibold text-foreground">
                  {result.latency_ms.toFixed(2)} ms
                </div>
              </div>
            </div>

            {result.explanation && (
              <div className="p-4 bg-slate-100 dark:bg-slate-800 rounded-lg">
                <div className="text-sm font-medium mb-2">Model Explanation</div>
                <pre className="text-xs overflow-auto max-h-48 bg-slate-50 dark:bg-slate-950 p-3 rounded">
                  {JSON.stringify(result.explanation, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}
      </Card>

      {/* Counter Speech Section - Only shown if toxic content detected */}
      {result?.is_toxic && (
        <Card className="p-6 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-950 dark:to-indigo-950 border-blue-200 dark:border-blue-800 shadow-lg">
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <MessageSquare className="w-6 h-6 text-blue-600 dark:text-blue-400" />
              <h3 className="text-xl font-bold text-foreground">Counter Speech Suggestion</h3>
            </div>

            {isGeneratingCounter ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="w-8 h-8 animate-spin text-blue-600 dark:text-blue-400" />
                <span className="ml-3 text-muted-foreground">Generating counter speech...</span>
              </div>
            ) : counterSpeechError ? (
              <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
                <div className="flex items-start gap-2">
                  <AlertTriangle className="w-5 h-5 text-destructive flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="font-medium text-destructive">Failed to generate counter speech</p>
                    <p className="text-sm text-destructive/80 mt-1">{counterSpeechError}</p>
                  </div>
                </div>
              </div>
            ) : counterSpeech ? (
              <div className="space-y-4 animate-fade-in">
                <div className="p-4 bg-white dark:bg-gray-900 rounded-lg border border-blue-200 dark:border-blue-800">
                  <div className="text-sm text-muted-foreground mb-2">Suggested Response:</div>
                  <p className="text-base text-foreground leading-relaxed">
                    {counterSpeech.counter_speech}
                  </p>
                </div>

                <div className="grid md:grid-cols-2 gap-3">
                  <div className="p-3 bg-white dark:bg-slate-900 rounded-lg border border-blue-100 dark:border-blue-900">
                    <div className="text-xs text-muted-foreground mb-1">Model</div>
                    <div className="text-sm font-medium text-foreground">
                      {counterSpeech.model}
                    </div>
                  </div>

                  {counterSpeech.generation_config && (
                    <div className="p-3 bg-white dark:bg-slate-900 rounded-lg border border-blue-100 dark:border-blue-900">
                      <div className="text-xs text-muted-foreground mb-1">Beam Search</div>
                      <div className="text-sm font-medium text-foreground">
                        {counterSpeech.generation_config.num_beams} beams
                      </div>
                    </div>
                  )}
                </div>

                <div className="text-xs text-muted-foreground italic">
                  This counter speech was automatically generated using IndoT5 to help respond constructively to toxic content.
                </div>
              </div>
            ) : null}
          </div>
        </Card>
      )}
    </div>
  );
};

export default ToxicityTest;