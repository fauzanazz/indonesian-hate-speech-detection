"use client";

import { SearchResult } from "@/types/dataset";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AlertTriangle, CheckCircle, Twitter, Instagram, FileText, Type } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";

interface SearchResultsProps {
  results: SearchResult[];
  searchQuery: string;
  isLoading: boolean;
  currentPage: number;
  pageSize: number;
}

function highlightText(text: string, query: string): React.ReactElement {
  if (!query.trim()) {
    return <span>{text}</span>;
  }

  const parts = text.split(new RegExp(`(${query})`, "gi"));
  
  return (
    <span>
      {parts.map((part, i) => (
        part.toLowerCase() === query.toLowerCase() ? (
          <mark key={i} className="bg-yellow-200 font-medium text-black">
            {part}
          </mark>
        ) : (
          <span key={i}>{part}</span>
        )
      ))}
    </span>
  );
}

function truncateText(text: string, query: string, maxLength: number = 200): string {
  if (text.length <= maxLength) return text;
  
  if (!query.trim()) {
    return text.substring(0, maxLength) + "...";
  }

  const lowerText = text.toLowerCase();
  const lowerQuery = query.toLowerCase();
  const index = lowerText.indexOf(lowerQuery);
  
  if (index === -1) {
    return text.substring(0, maxLength) + "...";
  }

  const start = Math.max(0, index - 50);
  const end = Math.min(text.length, index + query.length + 150);
  
  let result = text.substring(start, end);
  if (start > 0) result = "..." + result;
  if (end < text.length) result = result + "...";
  
  return result;
}

export function SearchResults({
  results,
  searchQuery,
  isLoading,
  currentPage,
  pageSize,
}: SearchResultsProps) {
  if (isLoading) {
    return (
      <div className="space-y-3">
        {Array.from({ length: pageSize }).map((_, i) => (
          <Card key={i}>
            <CardContent className="p-4">
              <div className="space-y-3">
                <div className="flex gap-2">
                  <Skeleton className="h-5 w-24" />
                  <Skeleton className="h-5 w-20" />
                </div>
                <Skeleton className="h-20 w-full" />
                <div className="flex gap-4">
                  <Skeleton className="h-4 w-24" />
                  <Skeleton className="h-4 w-24" />
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  if (results.length === 0) {
    return (
      <Alert>
        <FileText className="h-4 w-4" />
        <AlertDescription>
          No results found. Try adjusting your search query or filters.
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="space-y-3">
      {results.map((result, index) => {
        const displayText = truncateText(result.text, searchQuery);
        const resultNumber = (currentPage - 1) * pageSize + index + 1;
        
        return (
          <Card key={result.id} className="hover:shadow-md transition-shadow">
            <CardContent className="p-4">
              <div className="space-y-3">
                {/* Header with badges */}
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-xs text-muted-foreground">#{resultNumber}</span>
                  
                  <Badge
                    variant={result.labels === 1 ? "destructive" : "default"}
                    className="gap-1"
                  >
                    {result.labels === 1 ? (
                      <>
                        <AlertTriangle className="h-3 w-3" />
                        Hate Speech
                      </>
                    ) : (
                      <>
                        <CheckCircle className="h-3 w-3" />
                        Non-Hate Speech
                      </>
                    )}
                  </Badge>

                  <Badge variant="outline" className="gap-1">
                    {result.platform === "Twitter" ? (
                      <>
                        <Twitter className="h-3 w-3" />
                        Twitter
                      </>
                    ) : (
                      <>
                        <Instagram className="h-3 w-3" />
                        Instagram
                      </>
                    )}
                  </Badge>
                </div>

                {/* Text content with highlighting */}
                <div className="text-sm leading-relaxed">
                  {highlightText(displayText, searchQuery)}
                </div>

                {/* Statistics */}
                <div className="flex gap-4 text-xs text-muted-foreground">
                  <span className="flex items-center gap-1">
                    <FileText className="h-3 w-3" />
                    {result.char_count} chars
                  </span>
                  <span className="flex items-center gap-1">
                    <Type className="h-3 w-3" />
                    {result.word_count} words
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}