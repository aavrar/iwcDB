import React, { useState } from 'react';
import Header from '../components/Header';
import TimelineChart from '../components/TimelineChart';
import { ComparisonData, TimelinePoint } from '../types';
import ApiService from '../services/api';

interface CompareProps {
  onBackToHome: () => void;
}

const Compare: React.FC<CompareProps> = ({ onBackToHome }) => {
  const [wrestler1, setWrestler1] = useState('');
  const [wrestler2, setWrestler2] = useState('');
  const [comparisonData, setComparisonData] = useState<ComparisonData | null>(null);
  const [timeline1, setTimeline1] = useState<TimelinePoint[]>([]);
  const [timeline2, setTimeline2] = useState<TimelinePoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleCompare = async () => {
    if (!wrestler1.trim() || !wrestler2.trim()) {
      setError('Please enter both wrestler names to compare');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      
      // Fetch comparison data and timelines in parallel
      const [comparison, timeline1Data, timeline2Data] = await Promise.all([
        ApiService.compareWrestlers(wrestler1.trim(), wrestler2.trim()),
        ApiService.getTimeline(wrestler1.trim(), '30d'),
        ApiService.getTimeline(wrestler2.trim(), '30d')
      ]);
      
      setComparisonData(comparison);
      setTimeline1(timeline1Data);
      setTimeline2(timeline2Data);
    } catch (err) {
      setError('Failed to compare wrestlers. Please try again.');
      console.error('Compare error:', err);
    } finally {
      setLoading(false);
    }
  };

  const getSentimentColor = (score: number) => {
    if (score > 0.1) return 'text-wrestling-green';
    if (score < -0.1) return 'text-wrestling-red';
    return 'text-wrestling-gray';
  };

  const getDifferenceIndicator = (difference: number) => {
    if (Math.abs(difference) < 0.05) return '≈';
    return difference > 0 ? '↑' : '↓';
  };

  const renderWrestlerCard = (wrestler: any, isWinner: boolean, comparisonTitle: string) => (
    <div className={`wrestling-card ${isWinner ? 'border-wrestling-green' : ''}`}>
      {isWinner && (
        <div className="bg-wrestling-green text-black text-xs font-bold px-2 py-1 rounded mb-3 text-center">
          {comparisonTitle} WINNER
        </div>
      )}
      
      {/* Wrestler Image */}
      <div className="flex items-center gap-4 mb-4">
        <div className="w-16 h-16 bg-wrestling-black border-2 border-wrestling-red rounded overflow-hidden flex-shrink-0">
          {wrestler.image_url ? (
            <img
              src={wrestler.image_url}
              alt={wrestler.name}
              className="w-full h-full object-cover"
              onError={(e) => {
                const target = e.target as HTMLImageElement;
                target.style.display = 'none';
                const parent = target.parentElement;
                if (parent) {
                  const initials = wrestler.name.split(' ').map((n: string) => n[0]).join('').slice(0, 2);
                  parent.innerHTML = `<div class="text-wrestling-red text-lg font-bold w-full h-full flex items-center justify-center">${initials}</div>`;
                }
              }}
            />
          ) : (
            <div className="text-wrestling-red text-lg font-bold w-full h-full flex items-center justify-center">
              {wrestler.name.split(' ').map((n: string) => n[0]).join('').slice(0, 2)}
            </div>
          )}
        </div>
        
        <div>
          <h3 className="text-xl font-bold text-white uppercase tracking-wider">
            {wrestler.name}
          </h3>
          <p className="text-wrestling-gray text-sm">
            {wrestler.total_posts} posts analyzed
          </p>
        </div>
      </div>
      
      {/* Stats */}
      <div className="space-y-3">
        <div className="flex justify-between">
          <span className="text-wrestling-gray">Sentiment Score:</span>
          <span className={`font-bold ${getSentimentColor(wrestler.average_sentiment)}`}>
            {wrestler.average_sentiment > 0 ? '+' : ''}{wrestler.average_sentiment.toFixed(2)}
          </span>
        </div>
        
        <div className="flex justify-between">
          <span className="text-wrestling-gray">Popularity Score:</span>
          <span className="text-white font-bold">
            {wrestler.popularity_score.toFixed(2)}
          </span>
        </div>
        
        <div className="flex justify-between">
          <span className="text-wrestling-gray">Love Score:</span>
          <span className="text-wrestling-green font-bold">
            {wrestler.love_score.toFixed(2)}
          </span>
        </div>
        
        <div className="flex justify-between">
          <span className="text-wrestling-gray">Hate Score:</span>
          <span className="text-wrestling-red font-bold">
            {wrestler.hate_score.toFixed(2)}
          </span>
        </div>
        
        <div className="flex justify-between">
          <span className="text-wrestling-gray">Positive Posts:</span>
          <span className="text-wrestling-green font-bold">
            {wrestler.positive_posts}
          </span>
        </div>
        
        <div className="flex justify-between">
          <span className="text-wrestling-gray">Negative Posts:</span>
          <span className="text-wrestling-red font-bold">
            {wrestler.negative_posts}
          </span>
        </div>
      </div>
    </div>
  );

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

      {/* Compare Section */}
      <section className="bg-wrestling-charcoal p-8">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl font-black text-white uppercase tracking-wider text-center mb-6">
            COMPARE WRESTLERS
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <input
              type="text"
              value={wrestler1}
              onChange={(e) => setWrestler1(e.target.value)}
              placeholder="Enter first wrestler name..."
              className="wrestling-input text-lg"
              disabled={loading}
            />
            <input
              type="text"
              value={wrestler2}
              onChange={(e) => setWrestler2(e.target.value)}
              placeholder="Enter second wrestler name..."
              className="wrestling-input text-lg"
              disabled={loading}
            />
          </div>
          
          <div className="text-center">
            <button
              onClick={handleCompare}
              disabled={loading || !wrestler1.trim() || !wrestler2.trim()}
              className="wrestling-button text-xl px-8 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'COMPARING...' : 'COMPARE'}
            </button>
          </div>
          
          {error && (
            <div className="mt-4 p-4 bg-wrestling-red bg-opacity-20 border border-wrestling-red rounded text-center">
              <p className="text-wrestling-red font-bold">{error}</p>
            </div>
          )}
        </div>
      </section>

      {/* Comparison Results */}
      {comparisonData && (
        <>
          {/* Head-to-Head Stats */}
          <section className="p-8">
            <div className="max-w-7xl mx-auto">
              <h3 className="text-2xl font-black text-white uppercase tracking-wider text-center mb-6">
                HEAD-TO-HEAD COMPARISON
              </h3>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {renderWrestlerCard(
                  comparisonData.wrestler1, 
                  comparisonData.comparison.sentiment_difference > 0,
                  'SENTIMENT'
                )}
                {renderWrestlerCard(
                  comparisonData.wrestler2, 
                  comparisonData.comparison.sentiment_difference < 0,
                  'SENTIMENT'
                )}
              </div>
              
              {/* Comparison Summary */}
              <div className="mt-8 wrestling-card">
                <h4 className="text-xl font-bold text-white mb-4 text-center">COMPARISON SUMMARY</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-wrestling-gray">Sentiment Difference:</span>
                      <span className={`font-bold ${getSentimentColor(comparisonData.comparison.sentiment_difference)}`}>
                        {getDifferenceIndicator(comparisonData.comparison.sentiment_difference)} {Math.abs(comparisonData.comparison.sentiment_difference).toFixed(2)}
                      </span>
                    </div>
                    <div className="flex justify-between mb-2">
                      <span className="text-wrestling-gray">Popularity Difference:</span>
                      <span className="text-white font-bold">
                        {getDifferenceIndicator(comparisonData.comparison.popularity_difference)} {Math.abs(comparisonData.comparison.popularity_difference).toFixed(2)}
                      </span>
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-wrestling-gray">Love Difference:</span>
                      <span className="text-wrestling-green font-bold">
                        {getDifferenceIndicator(comparisonData.comparison.love_difference)} {Math.abs(comparisonData.comparison.love_difference).toFixed(2)}
                      </span>
                    </div>
                    <div className="flex justify-between mb-2">
                      <span className="text-wrestling-gray">Hate Difference:</span>
                      <span className="text-wrestling-red font-bold">
                        {getDifferenceIndicator(comparisonData.comparison.hate_difference)} {Math.abs(comparisonData.comparison.hate_difference).toFixed(2)}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Timeline Comparison */}
          <section className="p-8">
            <div className="max-w-7xl mx-auto">
              <h3 className="text-2xl font-black text-white uppercase tracking-wider text-center mb-6">
                SENTIMENT TIMELINE COMPARISON
              </h3>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <TimelineChart 
                  data={timeline1}
                  wrestlerName={comparisonData.wrestler1.name}
                />
                <TimelineChart 
                  data={timeline2}
                  wrestlerName={comparisonData.wrestler2.name}
                />
              </div>
            </div>
          </section>
        </>
      )}
    </div>
  );
};

export default Compare;