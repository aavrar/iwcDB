import React, { useState, useEffect } from 'react';
import ApiService from '../services/api';

interface TrendingWrestler {
  name: string;
  current_sentiment: number;
  previous_sentiment: number;
  sentiment_change: number;
  current_posts: number;
  image_url?: string;
  trend_strength: number;
}

interface TrendingData {
  rising_stars: TrendingWrestler[];
  falling_stars: TrendingWrestler[];
  period: string;
}

interface TrendingSectionProps {
  onWrestlerClick: (wrestlerName: string) => void;
}

const TrendingSection: React.FC<TrendingSectionProps> = ({ onWrestlerClick }) => {
  const [trendingData, setTrendingData] = useState<TrendingData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadTrendingData();
  }, []);

  const loadTrendingData = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await ApiService.getTrendingWrestlers(6);
      setTrendingData(data);
    } catch (err) {
      setError('Failed to load trending data');
      console.error('Trending data loading error:', err);
    } finally {
      setLoading(false);
    }
  };

  const getSentimentColor = (score: number) => {
    if (score > 0.1) return 'text-wrestling-green';
    if (score < -0.1) return 'text-wrestling-red';
    return 'text-wrestling-gray';
  };

  const getTrendIcon = (change: number) => {
    if (change > 0) return '↗';
    if (change < 0) return '↘';
    return '→';
  };

  const renderWrestlerCard = (wrestler: TrendingWrestler, isRising: boolean) => (
    <div 
      key={wrestler.name}
      onClick={() => onWrestlerClick(wrestler.name)}
      className="wrestling-card hover:border-wrestling-red transition-all duration-200 cursor-pointer p-4"
    >
      <div className="flex items-center gap-4">
        {/* Wrestler Image with Sentiment Frame */}
        <div className={`wrestler-image-container bg-wrestling-charcoal border-2 flex items-center justify-center ${
          wrestler.current_sentiment > 0.1 ? 'border-wrestling-green' : 
          wrestler.current_sentiment < -0.1 ? 'border-wrestling-red' : 
          'border-wrestling-gray'
        }`}>
          {wrestler.image_url ? (
            <img
              src={wrestler.image_url}
              alt={wrestler.name}
              className="wrestler-image"
              onError={(e) => {
                const target = e.target as HTMLImageElement;
                target.style.display = 'none';
                const parent = target.parentElement;
                if (parent) {
                  const initials = wrestler.name.split(' ').map(n => n[0]).join('').slice(0, 2);
                  parent.innerHTML = `<div class="text-wrestling-red text-xs text-center font-bold" style="width: 64px; height: 64px; display: flex; align-items: center; justify-content: center;">${initials}</div>`;
                }
              }}
            />
          ) : (
            <div className="text-wrestling-red text-xs text-center font-bold" style={{width: '64px', height: '64px', display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
              {wrestler.name.split(' ').map(n => n[0]).join('').slice(0, 2)}
            </div>
          )}
        </div>
        
        {/* Wrestler Info */}
        <div className="flex-1">
          <h4 className="text-white font-bold text-lg mb-1">
            {wrestler.name}
          </h4>
          <div className="flex items-center gap-4 text-sm">
            <span className="text-wrestling-gray">
              {wrestler.current_posts} posts
            </span>
            <span className={`font-bold ${getSentimentColor(wrestler.current_sentiment)}`}>
              {wrestler.current_sentiment > 0 ? '+' : ''}{wrestler.current_sentiment.toFixed(2)}
            </span>
            <span className="text-wrestling-gray">
              Previous: {wrestler.previous_sentiment > 0 ? '+' : ''}{wrestler.previous_sentiment.toFixed(2)}
            </span>
          </div>
        </div>
        
        {/* Trend Indicator */}
        <div className="text-right flex flex-col items-end">
          <div className={`text-2xl ${isRising ? 'text-wrestling-green' : 'text-wrestling-red'}`}>
            {getTrendIcon(wrestler.sentiment_change)}
          </div>
          <div className={`text-sm font-bold ${isRising ? 'text-wrestling-green' : 'text-wrestling-red'}`}>
            {wrestler.sentiment_change > 0 ? '+' : ''}{wrestler.sentiment_change.toFixed(2)}
          </div>
          <div className="text-xs text-wrestling-gray">
            #{Math.round(wrestler.trend_strength * 10)}
          </div>
        </div>
      </div>
    </div>
  );

  if (loading) {
    return (
      <section className="bg-wrestling-charcoal p-8">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-4 h-4 bg-wrestling-green"></div>
            <h3 className="text-2xl font-black text-white uppercase tracking-wider">
              TRENDING WRESTLERS
            </h3>
          </div>
          <div className="text-center text-wrestling-gray py-8">
            <div className="text-xl font-bold">LOADING TRENDS...</div>
          </div>
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section className="bg-wrestling-charcoal p-8">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-4 h-4 bg-wrestling-green"></div>
            <h3 className="text-2xl font-black text-white uppercase tracking-wider">
              TRENDING WRESTLERS
            </h3>
          </div>
          <div className="text-center text-wrestling-red py-8">
            <div className="text-xl font-bold">{error}</div>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="bg-wrestling-charcoal p-8">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-4 h-4 bg-wrestling-green"></div>
          <h3 className="text-2xl font-black text-white uppercase tracking-wider">
            TRENDING WRESTLERS
          </h3>
        </div>
        
        <p className="text-wrestling-gray text-sm mb-6 text-center">
          {trendingData?.period || 'Sentiment changes over time'}
        </p>
        
        {trendingData && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Rising Stars */}
            <div>
              <div className="flex items-center gap-2 mb-4">
                <div className="text-wrestling-green text-xl">↗</div>
                <h4 className="text-xl font-bold text-wrestling-green uppercase">
                  RISING STARS
                </h4>
              </div>
              
              {trendingData.rising_stars.length === 0 ? (
                <div className="text-center text-wrestling-gray py-4">
                  <p>No significant rising trends this period</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {trendingData.rising_stars.map(wrestler => 
                    renderWrestlerCard(wrestler, true)
                  )}
                </div>
              )}
            </div>
            
            {/* Falling Stars */}
            <div>
              <div className="flex items-center gap-2 mb-4">
                <div className="text-wrestling-red text-xl">↘</div>
                <h4 className="text-xl font-bold text-wrestling-red uppercase">
                  FALLING STARS
                </h4>
              </div>
              
              {trendingData.falling_stars.length === 0 ? (
                <div className="text-center text-wrestling-gray py-4">
                  <p>No significant falling trends this period</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {trendingData.falling_stars.map(wrestler => 
                    renderWrestlerCard(wrestler, false)
                  )}
                </div>
              )}
            </div>
          </div>
        )}
        
        {/* Refresh Button */}
        <div className="text-center mt-6">
          <button 
            onClick={loadTrendingData}
            className="wrestling-button text-sm"
            disabled={loading}
          >
            {loading ? 'REFRESHING...' : 'REFRESH TRENDS'}
          </button>
        </div>
      </div>
    </section>
  );
};

export default TrendingSection;