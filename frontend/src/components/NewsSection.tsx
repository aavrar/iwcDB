import React, { useState, useEffect } from 'react';
import ApiService from '../services/api';

interface NewsItem {
  id: string;
  title: string;
  content: string;
  subreddit: string;
  score: number;
  url: string;
  created_at: string;
  author: string;
  sentiment_score: number;
  sentiment_color: 'positive' | 'negative' | 'neutral';
  time_ago: string;
}

const NewsSection: React.FC = () => {
  const [newsItems, setNewsItems] = useState<NewsItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadNews();
  }, []);

  const loadNews = async () => {
    try {
      setLoading(true);
      setError(null);
      const news = await ApiService.getRecentNews(8);
      setNewsItems(news);
    } catch (err) {
      setError('Failed to load recent news');
      console.error('News loading error:', err);
    } finally {
      setLoading(false);
    }
  };

  const getSentimentBadgeClass = (color: string) => {
    switch (color) {
      case 'positive': return 'bg-wrestling-green text-black';
      case 'negative': return 'bg-wrestling-red text-white';
      default: return 'bg-wrestling-gray text-white';
    }
  };

  const getSubredditColor = (subreddit: string) => {
    const colors = {
      'SquaredCircle': 'text-blue-400',
      'WWE': 'text-yellow-400',
      'AEWOfficial': 'text-red-400',
      'njpw': 'text-purple-400',
      'ROH': 'text-green-400',
      'Wreddit': 'text-orange-400'
    };
    return colors[subreddit as keyof typeof colors] || 'text-wrestling-gray';
  };

  if (loading) {
    return (
      <section className="bg-wrestling-charcoal p-8">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-4 h-4 bg-wrestling-red"></div>
            <h3 className="text-2xl font-black text-white uppercase tracking-wider">
              LATEST NEWS & RUMORS
            </h3>
          </div>
          <div className="text-center text-wrestling-gray py-8">
            <div className="text-xl font-bold">LOADING NEWS...</div>
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
            <div className="w-4 h-4 bg-wrestling-red"></div>
            <h3 className="text-2xl font-black text-white uppercase tracking-wider">
              LATEST NEWS & RUMORS
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
          <div className="w-4 h-4 bg-wrestling-red"></div>
          <h3 className="text-2xl font-black text-white uppercase tracking-wider">
            LATEST NEWS & RUMORS
          </h3>
        </div>
        
        {newsItems.length === 0 ? (
          <div className="text-center text-wrestling-gray py-8">
            <div className="text-lg">No recent news items found</div>
            <div className="text-sm mt-2">Check back later for the latest wrestling updates</div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-4">
            {newsItems.map((item) => (
              <div key={item.id} className="wrestling-card hover:border-wrestling-red transition-all duration-200">
                {/* Header with subreddit and time */}
                <div className="flex justify-between items-center mb-3">
                  <span className={`text-sm font-bold ${getSubredditColor(item.subreddit)}`}>
                    r/{item.subreddit}
                  </span>
                  <span className="text-xs text-wrestling-gray">
                    {item.time_ago}
                  </span>
                </div>
                
                {/* Title */}
                <h4 className="text-white font-bold text-sm mb-2 line-clamp-2">
                  {item.title}
                </h4>
                
                {/* Content Preview */}
                <p className="text-wrestling-gray text-xs mb-3 line-clamp-2">
                  {item.content}
                </p>
                
                {/* Footer with sentiment and score */}
                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-2">
                    <span className={`text-xs px-2 py-1 rounded font-bold ${getSentimentBadgeClass(item.sentiment_color)}`}>
                      {item.sentiment_color.toUpperCase()}
                    </span>
                    <span className="text-xs text-wrestling-gray">
                      ↑ {item.score}
                    </span>
                  </div>
                  
                  <a 
                    href={item.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-wrestling-red hover:text-white transition-colors duration-200 font-bold"
                  >
                    READ MORE →
                  </a>
                </div>
              </div>
            ))}
          </div>
        )}
        
        {/* Refresh Button */}
        <div className="text-center mt-6">
          <button 
            onClick={loadNews}
            className="wrestling-button text-sm"
            disabled={loading}
          >
            {loading ? 'REFRESHING...' : 'REFRESH NEWS'}
          </button>
        </div>
      </div>
    </section>
  );
};

export default NewsSection;