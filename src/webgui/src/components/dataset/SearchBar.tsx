"use client";

import { Search, X } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { SearchFilters } from "@/types/dataset";

interface SearchBarProps {
  searchQuery: string;
  onSearchChange: (query: string) => void;
  filters: SearchFilters;
  onFilterChange: (filters: SearchFilters) => void;
  resultCount: number;
  isLoading: boolean;
}

export function SearchBar({
  searchQuery,
  onSearchChange,
  filters,
  onFilterChange,
  resultCount,
  isLoading,
}: SearchBarProps) {
  const handleClear = () => {
    onSearchChange("");
    onFilterChange({});
  };

  const hasActiveFilters = filters.label !== undefined || filters.platform !== undefined;

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            type="text"
            placeholder="Search dataset by text content..."
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            className="pl-10 pr-10"
          />
          {searchQuery && (
            <button
              onClick={() => onSearchChange("")}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>

        <Select
          value={filters.label?.toString() || "all"}
          onValueChange={(value: string) =>
            onFilterChange({
              ...filters,
              label: value === "all" ? undefined : (parseInt(value) as 0 | 1),
            })
          }
        >
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="Filter by label" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Labels</SelectItem>
            <SelectItem value="1">Hate Speech</SelectItem>
            <SelectItem value="0">Non-Hate Speech</SelectItem>
          </SelectContent>
        </Select>

        <Select
          value={filters.platform || "all"}
          onValueChange={(value: string) =>
            onFilterChange({
              ...filters,
              platform: value === "all" ? undefined : (value as "Twitter" | "Instagram"),
            })
          }
        >
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="Filter by platform" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Platforms</SelectItem>
            <SelectItem value="Twitter">Twitter</SelectItem>
            <SelectItem value="Instagram">Instagram</SelectItem>
          </SelectContent>
        </Select>

        {hasActiveFilters && (
          <Button variant="outline" onClick={handleClear} size="icon">
            <X className="h-4 w-4" />
          </Button>
        )}
      </div>

      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        {isLoading ? (
          <span>Searching...</span>
        ) : (
          <span>
            Found <strong className="text-foreground">{resultCount.toLocaleString()}</strong> results
          </span>
        )}

        {hasActiveFilters && (
          <div className="flex gap-1">
            {filters.label !== undefined && (
              <Badge variant="secondary">
                {filters.label === 1 ? "Hate Speech" : "Non-Hate Speech"}
              </Badge>
            )}
            {filters.platform && <Badge variant="secondary">{filters.platform}</Badge>}
          </div>
        )}
      </div>
    </div>
  );
}