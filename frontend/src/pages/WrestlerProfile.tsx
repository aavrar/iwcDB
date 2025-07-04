import React, { useState, useEffect } from 'react';
import Header from '../components/Header';
import TimelineChart from '../components/TimelineChart';
import TopPosts from '../components/TopPosts';
import { SearchResult, TimelinePoint } from '../types';
import ApiService from '../services/api';

interface WrestlerProfileProps {
  wrestlerName: string;
  onBackToHome: () => void;
  onSearch: (query: string) => void;
}

const WrestlerProfile: React.FC<WrestlerProfileProps> = ({ wrestlerName, onBackToHome, onSearch }) => {
  const [profileData, setProfileData] = useState<SearchResult | null>(null);
  const [timelineData, setTimelineData] = useState<TimelinePoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedPeriod, setSelectedPeriod] = useState<'30d' | '90d' | '6m' | '1y'>('30d');

  useEffect(() => {
    if (wrestlerName) {
      loadProfile(selectedPeriod);
    }
  }, [wrestlerName, selectedPeriod]);

  const loadProfile = async (period: string) => {
    try {
      setLoading(true);
      setError(null);
      
      const [profile, timeline] = await Promise.all([
        ApiService.search(wrestlerName, period),
        ApiService.getTimeline(wrestlerName, period)
      ]);
      
      console.log('Timeline data received:', timeline);
      console.log('Timeline data length:', timeline?.length);
      
      setProfileData(profile);
      setTimelineData(timeline);
    } catch (err) {
      setError('Failed to load wrestler profile');
      console.error('Profile loading error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handlePeriodChange = (period: '30d' | '90d' | '6m' | '1y') => {
    setSelectedPeriod(period);
  };

  const getSentimentColor = (score: number) => {
    if (score > 0.1) return 'text-wrestling-green';
    if (score < -0.1) return 'text-wrestling-red';
    return 'text-wrestling-gray';
  };

  const getSentimentLabel = (score: number) => {
    if (score > 0.3) return 'Very Positive';
    if (score > 0.1) return 'Positive';
    if (score > -0.1) return 'Neutral';
    if (score > -0.3) return 'Negative';
    return 'Very Negative';
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-wrestling-black">
        <Header />
        <div className="flex items-center justify-center h-96">
          <div className="text-2xl font-bold text-wrestling-red">LOADING PROFILE...</div>
        </div>
      </div>
    );
  }

  if (error || !profileData) {
    return (
      <div className="min-h-screen bg-wrestling-black">
        <Header />
        <div className="flex items-center justify-center h-96">
          <div className="text-center">
            <div className="text-2xl font-bold text-wrestling-red mb-4">
              {error || 'Profile not found'}
            </div>
            <button onClick={onBackToHome} className="wrestling-button">
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

      {/* Wrestler Profile Header */}
      <section className="bg-wrestling-charcoal p-8 border-b-2 border-wrestling-red">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-center gap-8 mb-6">
            {/* Wrestler Image with Sentiment Frame */}
            <div className={`w-32 h-32 bg-wrestling-black border-4 overflow-hidden rounded-lg flex-shrink-0 ${
              profileData.sentiment_summary.overall_sentiment > 0.1 ? 'border-wrestling-green' : 
              profileData.sentiment_summary.overall_sentiment < -0.1 ? 'border-wrestling-red' : 
              'border-wrestling-gray'
            }`}>
              {profileData.wrestler_image ? (
                <img
                  src={profileData.wrestler_image}
                  alt={wrestlerName}
                  className="w-full h-full object-cover object-center"
                  onError={(e) => {
                    const target = e.target as HTMLImageElement;
                    target.style.display = 'none';
                    const parent = target.parentElement;
                    if (parent) {
                      const initials = wrestlerName.split(' ').map(n => n[0]).join('').slice(0, 2);
                      parent.innerHTML = `<div class="text-wrestling-red text-2xl text-center font-bold" style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;">${initials}</div>`;
                    }
                  }}
                />
              ) : (
                <div className="text-wrestling-red text-2xl text-center font-bold w-full h-full flex items-center justify-center">
                  {wrestlerName.split(' ').map(n => n[0]).join('').slice(0, 2)}
                </div>
              )}
            </div>
            
            {/* Wrestler Info */}
            <div className="text-center flex-1">
              <h1 className="text-4xl font-black text-white uppercase tracking-wider mb-2">
                {wrestlerName}
              </h1>
              <p className="text-wrestling-gray text-lg mb-4">
                WRESTLING SENTIMENT PROFILE
              </p>
              
              {/* Key Stats */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-2xl mx-auto">
                <div className="text-center">
                  <div className="text-3xl font-bold text-wrestling-gray mb-1">
                    {profileData.sentiment_summary.total_posts}
                  </div>
                  <div className="text-sm text-wrestling-gray uppercase">
                    Posts Analyzed
                  </div>
                </div>
                
                <div className="text-center">
                  <div className={`text-3xl font-bold mb-1 ${getSentimentColor(profileData.sentiment_summary.overall_sentiment)}`}>
                    {profileData.sentiment_summary.overall_sentiment > 0 ? '+' : ''}
                    {profileData.sentiment_summary.overall_sentiment.toFixed(2)}
                  </div>
                  <div className="text-sm text-wrestling-gray uppercase">
                    Overall Sentiment
                  </div>
                </div>
                
                <div className="text-center">
                  <div className={`text-lg font-bold mb-1 ${getSentimentColor(profileData.sentiment_summary.overall_sentiment)}`}>
                    {getSentimentLabel(profileData.sentiment_summary.overall_sentiment)}
                  </div>
                  <div className="text-sm text-wrestling-gray uppercase">
                    Classification
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          {/* Period Selector */}
          <div className="flex justify-center gap-2">
            {(['30d', '90d', '6m', '1y'] as const).map((period) => (
              <button
                key={period}
                onClick={() => handlePeriodChange(period)}
                className={`px-4 py-2 text-sm font-bold uppercase transition-all duration-200 ${
                  selectedPeriod === period
                    ? 'bg-wrestling-red text-white'
                    : 'bg-wrestling-gray bg-opacity-20 text-wrestling-gray hover:bg-wrestling-red hover:text-white'
                }`}
              >
                {period === '30d' ? '30 Days' : 
                 period === '90d' ? '90 Days' : 
                 period === '6m' ? '6 Months' : '1 Year'}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Sentiment Breakdown */}
      <section className="p-8">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-2xl font-black text-white uppercase tracking-wider text-center mb-6">
            SENTIMENT BREAKDOWN
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="wrestling-card text-center">
              <div className="text-wrestling-green text-3xl font-bold mb-2">
                {profileData.sentiment_summary.positive_count}
              </div>
              <div className="text-white font-bold mb-1">POSITIVE POSTS</div>
              <div className="text-wrestling-gray text-sm">
                {((profileData.sentiment_summary.positive_count / profileData.sentiment_summary.total_posts) * 100).toFixed(1)}%
              </div>
            </div>
            
            <div className="wrestling-card text-center">
              <div className="text-wrestling-gray text-3xl font-bold mb-2">
                {profileData.sentiment_summary.neutral_count}
              </div>
              <div className="text-white font-bold mb-1">NEUTRAL POSTS</div>
              <div className="text-wrestling-gray text-sm">
                {((profileData.sentiment_summary.neutral_count / profileData.sentiment_summary.total_posts) * 100).toFixed(1)}%
              </div>
            </div>
            
            <div className="wrestling-card text-center">
              <div className="text-wrestling-red text-3xl font-bold mb-2">
                {profileData.sentiment_summary.negative_count}
              </div>
              <div className="text-white font-bold mb-1">NEGATIVE POSTS</div>
              <div className="text-wrestling-gray text-sm">
                {((profileData.sentiment_summary.negative_count / profileData.sentiment_summary.total_posts) * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Timeline Chart */}
      <section className="p-8">
        <div className="max-w-7xl mx-auto">
          <TimelineChart 
            data={timelineData}
            wrestlerName={wrestlerName}
          />
        </div>
      </section>

      {/* Top Posts */}
      <TopPosts 
        positivePost={profileData.top_positive_posts}
        negativePosts={profileData.top_negative_posts}
      />
    </div>
  );
};

export default WrestlerProfile;