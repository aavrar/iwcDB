export interface Post {
  id: string;
  post_id?: string;
  content: string;
  title?: string;
  score: number;
  url: string;
  source: string;
  subreddit?: string;
  created_at: string;
  author: string;
  sentiment_score?: number;
  image_url?: string;
  video_url?: string;
  content_type?: string;
  content_confidence?: number;
  sentiment?: string;
  sentiment_confidence?: number;
  needs_review?: boolean;
  manual_content_type?: string;
  manual_sentiment?: string;
}

export interface Wrestler {
  name: string;
  image_url?: string;
  total_posts: number;
  positive_posts: number;
  negative_posts: number;
  neutral_posts: number;
  average_sentiment: number;
  popularity_score: number;
  love_score: number;
  hate_score: number;
  recent_posts: Post[];
}

export interface SentimentSummary {
  overall_sentiment: number;
  total_posts: number;
  positive_count: number;
  negative_count: number;
  neutral_count: number;
  period: '30d' | '90d' | '6m' | '1y';
}

export interface TimelinePoint {
  date: string;
  sentiment: number;
  posts_count: number;
}

export interface SearchResult {
  query: string;
  type: 'wrestler' | 'event' | 'brand';
  wrestler_image?: string;
  sentiment_summary: SentimentSummary;
  top_positive_posts: Post[];
  top_negative_posts: Post[];
  timeline: TimelinePoint[];
  related_wrestlers?: Wrestler[];
}

export interface ComparisonData {
  wrestler1: Wrestler;
  wrestler2: Wrestler;
  comparison: {
    sentiment_difference: number;
    popularity_difference: number;
    love_difference: number;
    hate_difference: number;
  };
}

export interface PopularWrestler {
  name: string;
  image_url?: string;
  post_count: number;
  sentiment_score: number;
  rank: number;
}