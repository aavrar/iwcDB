import axios from 'axios';
import { SearchResult, PopularWrestler, ComparisonData, TimelinePoint } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

// Add request interceptor for debugging
api.interceptors.request.use(request => {
  console.log('Making API request to:', request.url);
  console.log('Full URL:', `${request.baseURL}${request.url}`);
  return request;
});

// Add response interceptor for debugging
api.interceptors.response.use(
  response => {
    console.log('API response received:', response.status, response.data);
    return response;
  },
  error => {
    console.error('API request failed:', error.message);
    console.error('Error details:', error.response?.data || error);
    return Promise.reject(error);
  }
);

export class ApiService {
  static async search(query: string, period: string = '30d'): Promise<SearchResult> {
    const response = await api.get(`/api/v1/search`, {
      params: { q: query, period }
    });
    return response.data;
  }

  static async getPopularWrestlers(): Promise<PopularWrestler[]> {
    const response = await api.get('/api/v1/popular');
    return response.data;
  }

  static async getMostLovedWrestlers(): Promise<PopularWrestler[]> {
    const response = await api.get('/api/v1/loved');
    return response.data;
  }

  static async getMostHatedWrestlers(): Promise<PopularWrestler[]> {
    const response = await api.get('/api/v1/hated');
    return response.data;
  }

  static async getTimeline(query: string, period: string = '30d'): Promise<TimelinePoint[]> {
    const response = await api.get('/api/v1/timeline', {
      params: { q: query, period }
    });
    return response.data;
  }

  static async compareWrestlers(wrestler1: string, wrestler2: string): Promise<ComparisonData> {
    const response = await api.get('/api/v1/compare', {
      params: { w1: wrestler1, w2: wrestler2 }
    });
    return response.data;
  }

  static async getTopPosts(type: 'positive' | 'negative', limit: number = 3): Promise<any[]> {
    const response = await api.get('/api/v1/posts/top', {
      params: { type, limit }
    });
    return response.data;
  }

  static async getRecentNews(limit: number = 10): Promise<any[]> {
    const response = await api.get('/api/v1/news', {
      params: { limit }
    });
    return response.data;
  }

  static async getWrestlerSuggestions(query: string, limit: number = 8): Promise<any[]> {
    if (query.length < 1) return [];
    
    const response = await api.get('/api/v1/suggestions', {
      params: { q: query, limit }
    });
    return response.data;
  }

  static async getTrendingWrestlers(limit: number = 10): Promise<any> {
    const response = await api.get('/api/v1/trending', {
      params: { limit }
    });
    return response.data;
  }

  static async getNextTrainingPost(): Promise<any> {
    const response = await api.get('/api/v1/training/next-post');
    return response.data;
  }

  static async classifyTrainingPost(classification: any): Promise<any> {
    const response = await api.post('/api/v1/training/classify', classification);
    return response.data;
  }

  static async saveTrainingModel(): Promise<any> {
    const response = await api.post('/api/v1/training/save-model');
    return response.data;
  }

  // Labeling interface methods
  static async makeRequest(endpoint: string, options?: {method?: string, body?: string}): Promise<any> {
    const { method = 'GET', body } = options || {};
    
    if (method === 'POST') {
      const response = await api.post(`/api/v1${endpoint}`, body ? JSON.parse(body) : {});
      return response.data;
    } else {
      const response = await api.get(`/api/v1${endpoint}`);
      return response.data;
    }
  }
}

export default ApiService;