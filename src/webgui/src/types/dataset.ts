export interface DatasetRecord {
  id: number;
  text: string;
  labels: 0 | 1;
  platform: 'Twitter' | 'Instagram';
  char_count: number;
  word_count: number;
}

export interface SearchResult extends DatasetRecord {
  score: number;
  highlights?: string[];
}

export interface SearchFilters {
  label?: 0 | 1;
  platform?: 'Twitter' | 'Instagram';
  minWords?: number;
  maxWords?: number;
}

export interface PaginationState {
  currentPage: number;
  pageSize: number;
  totalResults: number;
  totalPages: number;
}

export interface DatasetResponse {
  records: DatasetRecord[];
  total: number;
  page: number;
  pageSize: number;
  hasMore: boolean;
}