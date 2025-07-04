import React, { useState, useEffect } from 'react';
import Header from '../components/Header';
import SearchSection from '../components/SearchSection';
import SentimentSummary from '../components/SentimentSummary';
import TopPosts from '../components/TopPosts';
import TimelineChart from '../components/TimelineChart';
import { SearchResult, TimelinePoint } from '../types';
import ApiService from '../services/api';

interface SearchResultsProps {
  query: string;
  onSearch: (query: string) => void;
  onBackToHome: () => void;
}

const SearchResults: React.FC<SearchResultsProps> = ({ query, onSearch, onBackToHome }) => {
  const [results, setResults] = useState<SearchResult | null>(null);
  const [timelineData, setTimelineData] = useState<TimelinePoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (query) {
      performSearch(query);
    }
  }, [query]);

  const performSearch = async (searchQuery: string, period: string = '30d') => {
    try {
      setLoading(true);
      setError(null);
      
      // Fetch both search results and timeline data in parallel
      const [result, timeline] = await Promise.all([
        ApiService.search(searchQuery, period),
        ApiService.getTimeline(searchQuery, period)
      ]);
      
      setResults(result);
      setTimelineData(timeline);
    } catch (err) {
      setError('Failed to fetch search results. Please try again.');
      console.error('Search error:', err);
      
      // Mock data for development
      const mockResult: SearchResult = {
        query: searchQuery,
        type: 'wrestler',
        sentiment_summary: {
          overall_sentiment: 0.45,
          total_posts: 1234,
          positive_count: 567,
          negative_count: 234,
          neutral_count: 433,
          period: '30d' as const,
        },
        top_positive_posts: [
          {
            id: '1',
            content: `${searchQuery} had an absolutely incredible match tonight! The storytelling was phenomenal and the crowd was electric. This is why I love wrestling!`,
            title: `${searchQuery} delivers again!`,
            score: 1247,
            url: 'https://reddit.com/example',
            source: 'reddit',
            subreddit: 'SquaredCircle',
            created_at: new Date().toISOString(),
            author: 'WrestlingFan123',
            sentiment_score: 0.87
          },
          {
            id: '2',
            content: `That promo from ${searchQuery} was amazing. Great character work and really compelling stuff. Best wrestler on the roster right now.`,
            title: 'Promo of the year?',
            score: 892,
            url: 'https://reddit.com/example2',
            source: 'reddit',
            subreddit: 'WWE',
            created_at: new Date().toISOString(),
            author: 'PromoExpert',
            sentiment_score: 0.73
          }
        ],
        top_negative_posts: [
          {
            id: '3',
            content: `The booking for ${searchQuery} has been terrible lately. This storyline makes no sense and is completely ruining the character.`,
            title: 'Booking complaints',
            score: -156,
            url: 'https://reddit.com/example3',
            source: 'reddit',
            subreddit: 'SquaredCircle',
            created_at: new Date().toISOString(),
            author: 'BookingCritic',
            sentiment_score: -0.65
          }
        ],
        timeline: []
      };
      setResults(mockResult);
      setTimelineData([]);
    } finally {
      setLoading(false);
    }
  };

  const handlePeriodChange = (period: '30d' | '90d' | '6m' | '1y') => {
    if (results) {
      performSearch(query, period);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-wrestling-black">
        <Header />
        <SearchSection onSearch={onSearch} isLoading={true} />
        <div className="flex items-center justify-center h-96">
          <div className="text-center">
            <div className="text-3xl font-bold text-wrestling-red mb-4">ANALYZING...</div>
            <div className="text-wrestling-gray">Searching the wrestling community for "{query}"</div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-wrestling-black">
        <Header />
        <SearchSection onSearch={onSearch} />
        <div className="flex items-center justify-center h-96">
          <div className="text-center">
            <div className="text-2xl font-bold text-wrestling-red mb-4">ERROR</div>
            <div className="text-wrestling-gray mb-4">{error}</div>
            <button 
              onClick={onBackToHome}
              className="wrestling-button"
            >
              BACK TO HOME
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="min-h-screen bg-wrestling-black">
        <Header />
        <SearchSection onSearch={onSearch} />
        <div className="flex items-center justify-center h-96">
          <div className="text-center">
            <div className="text-2xl font-bold text-wrestling-gray mb-4">NO RESULTS</div>
            <button 
              onClick={onBackToHome}
              className="wrestling-button"
            >
              BACK TO HOME
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-wrestling-black">
      <Header />
      <SearchSection onSearch={onSearch} />
      
      {/* Back Button */}
      <div className="bg-wrestling-charcoal border-b border-wrestling-gray">
        <div className="max-w-7xl mx-auto p-4">
          <button 
            onClick={onBackToHome}
            className="wrestling-button"
          >
            ← BACK TO HOME
          </button>
        </div>
      </div>

      {/* Results Header with Wrestler Profile - Properly Centered Layout */}
      <section className="bg-wrestling-charcoal p-8 border-b-2 border-wrestling-red">
        <div className="max-w-7xl mx-auto">
          {/* Centered Profile Display - Flex Layout for Perfect Centering */}
          <div className="flex justify-center items-center mb-6">
            <div className={`w-32 h-32 bg-wrestling-black border-4 overflow-hidden rounded-lg ${
              results.sentiment_summary.overall_sentiment > 0.1 ? 'border-wrestling-green' : 
              results.sentiment_summary.overall_sentiment < -0.1 ? 'border-wrestling-red' : 
              'border-wrestling-gray'
            }`}>
              {results.wrestler_image ? (
                <img
                  src={results.wrestler_image}
                  alt={query}
                  className="w-full h-full object-cover object-center"
                  onError={(e) => {
                    const target = e.target as HTMLImageElement;
                    target.style.display = 'none';
                    const parent = target.parentElement;
                    if (parent) {
                      const initials = query.split(' ').map(n => n[0]).join('').slice(0, 2);
                      parent.innerHTML = `<div class="text-wrestling-red text-2xl text-center font-bold" style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;">${initials}</div>`;
                    }
                  }}
                />
              ) : (
                <div className="text-wrestling-red text-2xl text-center font-bold w-full h-full flex items-center justify-center">
                  {query.split(' ').map(n => n[0]).join('').slice(0, 2)}
                </div>
              )}
            </div>
          </div>
          
          {/* Wrestler Name and Info - Centered */}
          <div className="text-center">
            <h2 className="text-4xl font-black text-white uppercase tracking-wider mb-2">
              {query}
            </h2>
            <p className="text-wrestling-gray text-lg mb-4">
              {results.type.toUpperCase()} ANALYSIS
            </p>
            <div className="flex justify-center gap-12">
              <div className="text-center">
                <div className="text-3xl font-bold text-wrestling-gray">
                  {results.sentiment_summary.total_posts}
                </div>
                <div className="text-sm text-wrestling-gray uppercase">
                  Posts Analyzed
                </div>
              </div>
              <div className="text-center">
                <div className={`text-3xl font-bold ${
                  results.sentiment_summary.overall_sentiment > 0.1 
                    ? 'text-wrestling-green' 
                    : results.sentiment_summary.overall_sentiment < -0.1 
                    ? 'text-wrestling-red' 
                    : 'text-wrestling-gray'
                }`}>
                  {results.sentiment_summary.overall_sentiment > 0 ? '+' : ''}
                  {results.sentiment_summary.overall_sentiment.toFixed(2)}
                </div>
                <div className="text-sm text-wrestling-gray uppercase">
                  Overall Sentiment
                </div>
              </div>
              <div className="text-center">
                <div className={`text-lg font-bold ${
                  results.sentiment_summary.overall_sentiment > 0.1 
                    ? 'text-wrestling-green' 
                    : results.sentiment_summary.overall_sentiment < -0.1 
                    ? 'text-wrestling-red' 
                    : 'text-wrestling-gray'
                }`}>
                  {results.sentiment_summary.overall_sentiment > 0.3 ? 'Very Positive' :
                   results.sentiment_summary.overall_sentiment > 0.1 ? 'Positive' :
                   results.sentiment_summary.overall_sentiment > -0.1 ? 'Neutral' :
                   results.sentiment_summary.overall_sentiment > -0.3 ? 'Negative' : 'Very Negative'}
                </div>
                <div className="text-sm text-wrestling-gray uppercase">
                  Classification
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <SentimentSummary 
        summary={results.sentiment_summary}
        onPeriodChange={handlePeriodChange}
      />
      
      {/* Timeline Chart */}
      <section className="p-6">
        <div className="max-w-7xl mx-auto">
          <TimelineChart 
            data={timelineData}
            wrestlerName={query}
          />
        </div>
      </section>
      
      <TopPosts 
        positivePost={results.top_positive_posts}
        negativePosts={results.top_negative_posts}
      />
    </div>
  );
};

export default SearchResults;