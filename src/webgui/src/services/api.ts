const API_BASE = "http://localhost:8001/api/v1";

// Toxicity Detection Types
export interface ToxicityRequest {
  text: string;
  return_explanation?: boolean;
}

export interface ToxicityResponse {
  text: string;
  is_toxic: boolean;
  toxicity_score: number;
  confidence: string;
  tier: string;
  model_name: string;
  latency_ms: number;
  trace_id: string;
  explanation?: Record<string, unknown>;
}

// Search Types
export interface SearchRequest {
  query: string;
  top_k?: number;
  score_threshold?: number;
}

export interface SearchResultItem {
  text: string;
  score: number;
  metadata: Record<string, unknown>;
}

export interface SearchResponse {
  query: string;
  results: SearchResultItem[];
  count: number;
}

// Toxicity Detection Service
export const toxicityService = {
  async detectBasic(text: string, returnExplanation = false): Promise<ToxicityResponse> {
    const response = await fetch(`${API_BASE}/toxicity/basic`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, return_explanation: returnExplanation }),
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }
    
    return response.json();
  },

  async detectContextual(text: string, returnExplanation = false): Promise<ToxicityResponse> {
    const response = await fetch(`${API_BASE}/toxicity/contextual`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, return_explanation: returnExplanation }),
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }
    
    return response.json();
  },

  async detectSociolinguistic(text: string, returnExplanation = false): Promise<ToxicityResponse> {
    const response = await fetch(`${API_BASE}/toxicity/sociolinguistic`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, return_explanation: returnExplanation }),
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }
    
    return response.json();
  },

  async detectEnsemble(text: string, returnExplanation = false): Promise<ToxicityResponse> {
    const response = await fetch(`${API_BASE}/toxicity/ensemble`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, return_explanation: returnExplanation }),
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }
    
    return response.json();
  },
};

// Search Service
export const searchService = {
  async search(query: string, topK = 10, scoreThreshold = 0.7): Promise<SearchResponse> {
    const response = await fetch(`${API_BASE}/search/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 
        query, 
        top_k: topK, 
        score_threshold: scoreThreshold 
      }),
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }
    
    return response.json();
  },
};