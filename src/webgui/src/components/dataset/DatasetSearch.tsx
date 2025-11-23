"use client";

import { useState, useEffect, useMemo, useCallback } from "react";
import FlexSearch from "flexsearch";
import { SearchBar } from "./SearchBar";
import { SearchResults } from "./SearchResults";
import { PaginationControls } from "./PaginationControls";
import { DatasetRecord, SearchResult, SearchFilters, PaginationState } from "@/types/dataset";
import { Card, CardContent } from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { AlertCircle } from "lucide-react";

export function DatasetSearch() {
  const [allRecords, setAllRecords] = useState<DatasetRecord[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const [filters, setFilters] = useState<SearchFilters>({});
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pagination, setPagination] = useState<PaginationState>({
    currentPage: 1,
    pageSize: 20,
    totalResults: 0,
    totalPages: 0,
  });

  // FlexSearch index
  const searchIndex = useMemo(() => {
    const index = new FlexSearch.Index({
      preset: "score",
      tokenize: "full",
      resolution: 9,
      cache: true,
      context: {
        depth: 3,
        bidirectional: true,
      },
    });

    return index;
  }, []);

  // Fetch all records on mount
  useEffect(() => {
    const fetchRecords = async () => {
      try {
        setIsLoading(true);
        setError(null);

        // Fetch all records without pagination for client-side indexing
        const response = await fetch("/api/dataset?limit=100000");
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        
        if (data.error) {
          throw new Error(data.error);
        }

        setAllRecords(data.records);

        // Build FlexSearch index
        data.records.forEach((record: DatasetRecord) => {
          searchIndex.add(record.id, record.text);
        });

        setIsLoading(false);
      } catch (err) {
        console.error("Error fetching dataset:", err);
        setError(err instanceof Error ? err.message : "Failed to load dataset");
        setIsLoading(false);
      }
    };

    fetchRecords();
  }, [searchIndex]);

  // Debounce search query
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedQuery(searchQuery);
    }, 300);

    return () => clearTimeout(timer);
  }, [searchQuery]);

  // Perform search and filtering
  const searchResults = useMemo(() => {
    let results: SearchResult[] = [];

    if (!debouncedQuery.trim()) {
      // No search query - return all records
      results = allRecords.map((record) => ({
        ...record,
        score: 1,
      }));
    } else {
      // Perform FlexSearch
      const searchIds = searchIndex.search(debouncedQuery, { limit: 10000 });
      
      results = searchIds
        .map((id) => {
          const record = allRecords.find((r) => r.id === id);
          if (!record) return null;
          
          return {
            ...record,
            score: 1,
          };
        })
        .filter((r): r is SearchResult => r !== null);
    }

    // Apply filters
    if (filters.label !== undefined) {
      results = results.filter((r) => r.labels === filters.label);
    }

    if (filters.platform) {
      results = results.filter((r) => r.platform === filters.platform);
    }

    if (filters.minWords) {
      results = results.filter((r) => r.word_count >= filters.minWords!);
    }

    if (filters.maxWords) {
      results = results.filter((r) => r.word_count <= filters.maxWords!);
    }

    return results;
  }, [allRecords, debouncedQuery, filters, searchIndex]);

  // Update pagination when results change
  useEffect(() => {
    const totalResults = searchResults.length;
    const totalPages = Math.ceil(totalResults / pagination.pageSize);

    setPagination((prev) => ({
      ...prev,
      currentPage: 1, // Reset to first page
      totalResults,
      totalPages,
    }));
  }, [searchResults, pagination.pageSize]);

  // Get paginated results
  const paginatedResults = useMemo(() => {
    const startIdx = (pagination.currentPage - 1) * pagination.pageSize;
    const endIdx = startIdx + pagination.pageSize;
    return searchResults.slice(startIdx, endIdx);
  }, [searchResults, pagination.currentPage, pagination.pageSize]);

  const handlePageChange = useCallback((page: number) => {
    setPagination((prev) => ({ ...prev, currentPage: page }));
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  const handlePageSizeChange = useCallback((pageSize: number) => {
    setPagination((prev) => ({
      ...prev,
      pageSize,
      currentPage: 1,
      totalPages: Math.ceil(prev.totalResults / pageSize),
    }));
  }, []);

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>
          {error}
          <button
            onClick={() => window.location.reload()}
            className="ml-2 underline"
          >
            Retry
          </button>
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <Card>
      <CardContent className="p-6 space-y-6">
        <div>
          <h2 className="text-2xl font-semibold mb-2">Dataset Search</h2>
          <p className="text-sm text-muted-foreground">
            Search through {allRecords.length.toLocaleString()} records using full-text search
          </p>
        </div>

        <SearchBar
          searchQuery={searchQuery}
          onSearchChange={setSearchQuery}
          filters={filters}
          onFilterChange={setFilters}
          resultCount={pagination.totalResults}
          isLoading={isLoading}
        />

        <SearchResults
          results={paginatedResults}
          searchQuery={debouncedQuery}
          isLoading={isLoading}
          currentPage={pagination.currentPage}
          pageSize={pagination.pageSize}
        />

        {!isLoading && pagination.totalPages > 1 && (
          <PaginationControls
            pagination={pagination}
            onPageChange={handlePageChange}
            onPageSizeChange={handlePageSizeChange}
          />
        )}
      </CardContent>
    </Card>
  );
}