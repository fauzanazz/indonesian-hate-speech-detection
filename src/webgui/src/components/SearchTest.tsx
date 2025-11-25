"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Search, Loader2, AlertCircle } from "lucide-react";
import { searchService, SearchResponse } from "@/services/api";

const SearchTest = () => {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(10);
  const [scoreThreshold, setScoreThreshold] = useState(0.7);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const performSearch = async () => {
    if (!query.trim()) return;
    
    setIsSearching(true);
    setError(null);
    
    try {
      const response = await searchService.search(query, topK, scoreThreshold);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to perform search");
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <div className="space-y-6">
      <Card className="p-6 bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-700 shadow-lg">
        <div className="space-y-4">
          <div>
            <label className="text-sm font-medium mb-2 block">Search Query</label>
            <div className="flex gap-2">
              <Input
                placeholder="Enter text to search for similar toxic content..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && performSearch()}
                className="flex-1"
              />
              <Button 
                onClick={performSearch}
                disabled={!query.trim() || isSearching}
                className="bg-primary hover:bg-primary/90"
              >
                {isSearching ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Search className="w-4 h-4" />
                )}
              </Button>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <label className="text-sm font-medium mb-2 block">
                Top K Results: {topK}
              </label>
              <Slider
                value={[topK]}
                onValueChange={([value]: number[]) => setTopK(value)}
                min={1}
                max={50}
                step={1}
                className="w-full"
              />
            </div>

            <div>
              <label className="text-sm font-medium mb-2 block">
                Score Threshold: {scoreThreshold.toFixed(2)}
              </label>
              <Slider
                value={[scoreThreshold]}
                onValueChange={([value]: number[]) => setScoreThreshold(value)}
                min={0}
                max={1}
                step={0.05}
                className="w-full"
              />
            </div>
          </div>
        </div>
      </Card>

      <Card className="p-8 bg-white dark:bg-slate-900 border-slate-200 dark:border-slate-700 shadow-lg min-h-[200px]">
        {error ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center text-destructive">
              <AlertCircle className="w-12 h-12 mx-auto mb-4" />
              <p className="text-lg font-semibold mb-2">Error</p>
              <p className="text-sm">{error}</p>
            </div>
          </div>
        ) : !result ? (
          <div className="flex items-center justify-center h-full text-muted-foreground">
            <p className="text-lg">Search results will appear here</p>
          </div>
        ) : (
          <div className="animate-fade-in space-y-4">
            <div className="flex items-center justify-between pb-4 border-b">
              <h3 className="text-xl font-bold">Search Results</h3>
              <span className="text-sm text-muted-foreground">
                Found {result.count} similar {result.count === 1 ? "item" : "items"}
              </span>
            </div>

            {result.count === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <AlertCircle className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p className="text-lg">No similar content found</p>
                <p className="text-sm mt-2">Try adjusting the score threshold or search query</p>
              </div>
            ) : (
              <div className="space-y-3 max-h-[500px] overflow-y-auto">
                {result.results.map((item, index) => (
                  <Card key={index} className="p-4 bg-slate-50 dark:bg-slate-800 border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-750 transition-colors">
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1">
                        <p className="text-sm mb-2">{item.text}</p>
                        {Object.keys(item.metadata).length > 0 && (
                          <div className="flex flex-wrap gap-2 mt-2">
                            {Object.entries(item.metadata).map(([key, value]) => (
                              <span 
                                key={key}
                                className="px-2 py-1 bg-slate-100 dark:bg-slate-900 text-xs rounded border border-slate-200 dark:border-slate-600"
                              >
                                <span className="font-medium">{key}:</span> {String(value)}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                      <div className="flex flex-col items-end gap-1">
                        <span className="text-xs text-muted-foreground">Score</span>
                        <span className="text-lg font-bold text-primary">
                          {(item.score * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            )}
          </div>
        )}
      </Card>
    </div>
  );
};

export default SearchTest;