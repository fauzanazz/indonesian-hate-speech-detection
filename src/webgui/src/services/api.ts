const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8001/api/v1";

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

// Counter Speech Types
export interface CounterSpeechRequest {
  text: string;
  max_length?: number;
  num_beams?: number;
  temperature?: number;
  do_sample?: boolean;
}

export interface CounterSpeechResponse {
  text: string;
  counter_speech: string;
  model: string;
  generation_config?: {
    max_length: number;
    num_beams: number;
    temperature: number;
    do_sample: boolean;
    length_penalty: number;
    repetition_penalty: number;
  };
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

// Counter Speech Service
export const counterSpeechService = {
  async generate(
    text: string,
    options?: {
      maxLength?: number;
      numBeams?: number;
      temperature?: number;
      doSample?: boolean;
    }
  ): Promise<CounterSpeechResponse> {
    const response = await fetch(`${API_BASE}/counter-speech/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        max_length: options?.maxLength,
        num_beams: options?.numBeams,
        temperature: options?.temperature,
        do_sample: options?.doSample,
      }),
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }

    return response.json();
  },
};