import React from 'react';
import { SentimentSummary as SentimentSummaryType } from '../types';

interface SentimentSummaryProps {
  summary: SentimentSummaryType;
  onPeriodChange: (period: '30d' | '90d' | '6m' | '1y') => void;
}

const SentimentSummary: React.FC<SentimentSummaryProps> = ({ summary, onPeriodChange }) => {
  const getSentimentColor = (score: number): string => {
    if (score > 0.1) return 'sentiment-positive';
    if (score < -0.1) return 'sentiment-negative';
    return 'sentiment-neutral';
  };

  const getSentimentLabel = (score: number): string => {
    if (score > 0.3) return 'VERY POSITIVE';
    if (score > 0.1) return 'POSITIVE';
    if (score < -0.3) return 'VERY NEGATIVE';
    if (score < -0.1) return 'NEGATIVE';
    return 'NEUTRAL';
  };

  const formatScore = (score: number): string => {
    return (score >= 0 ? '+' : '') + score.toFixed(2);
  };

  return (
    <section className="bg-wrestling-charcoal p-8 border-b-4 border-wrestling-red">
      <div className="max-w-6xl mx-auto">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Main Sentiment Score */}
          <div className="text-center">
            <div className={`inline-block p-8 border-4 ${getSentimentColor(summary.overall_sentiment)}`}>
              <div className="text-6xl font-black mb-2">
                {formatScore(summary.overall_sentiment)}
              </div>
              <div className="text-2xl font-bold tracking-wider">
                {getSentimentLabel(summary.overall_sentiment)}
              </div>
            </div>
            
            <div className="mt-6 text-wrestling-gray">
              <div className="text-3xl font-bold">{summary.total_posts.toLocaleString()}</div>
              <div className="text-lg">POSTS ANALYZED</div>
            </div>
          </div>

          {/* Breakdown */}
          <div className="space-y-4">
            <div className="wrestling-card">
              <div className="flex justify-between items-center">
                <span className="text-wrestling-green font-bold">POSITIVE</span>
                <span className="text-2xl font-black">{summary.positive_count}</span>
              </div>
              <div className="bg-wrestling-green h-2 mt-2" style={{
                width: `${(summary.positive_count / summary.total_posts) * 100}%`
              }}></div>
            </div>

            <div className="wrestling-card">
              <div className="flex justify-between items-center">
                <span className="text-wrestling-red font-bold">NEGATIVE</span>
                <span className="text-2xl font-black">{summary.negative_count}</span>
              </div>
              <div className="bg-wrestling-red h-2 mt-2" style={{
                width: `${(summary.negative_count / summary.total_posts) * 100}%`
              }}></div>
            </div>

            <div className="wrestling-card">
              <div className="flex justify-between items-center">
                <span className="text-wrestling-gray font-bold">NEUTRAL</span>
                <span className="text-2xl font-black">{summary.neutral_count}</span>
              </div>
              <div className="bg-wrestling-gray h-2 mt-2" style={{
                width: `${(summary.neutral_count / summary.total_posts) * 100}%`
              }}></div>
            </div>
          </div>
        </div>

        {/* Period Controls */}
        <div className="mt-8 flex justify-center gap-2">
          {[
            { key: '30d' as const, label: '30 DAYS' },
            { key: '90d' as const, label: '3 MONTHS' },
            { key: '6m' as const, label: '6 MONTHS' },
            { key: '1y' as const, label: '1 YEAR' }
          ].map(({ key, label }) => (
            <button
              key={key}
              onClick={() => onPeriodChange(key)}
              className={`px-4 py-2 border-2 font-bold transition-all duration-200 ${
                summary.period === key
                  ? 'bg-wrestling-red text-white border-wrestling-red'
                  : 'bg-transparent text-wrestling-gray border-wrestling-gray hover:border-wrestling-red hover:text-white'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
    </section>
  );
};

export default SentimentSummary;