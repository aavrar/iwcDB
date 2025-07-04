import React from 'react';
import { PopularWrestler } from '../types';

interface WrestlerCardProps {
  wrestler: PopularWrestler;
  type: 'popular' | 'loved' | 'hated';
  onClick?: () => void;
}

const WrestlerCard: React.FC<WrestlerCardProps> = ({ wrestler, type, onClick }) => {
  const getTypeColor = () => {
    switch (type) {
      case 'loved': return 'border-wrestling-green';
      case 'hated': return 'border-wrestling-red';
      default: return 'border-wrestling-gray';
    }
  };

  const getScoreColor = () => {
    if (wrestler.sentiment_score > 0.1) return 'text-wrestling-green';
    if (wrestler.sentiment_score < -0.1) return 'text-wrestling-red';
    return 'text-wrestling-gray';
  };

  return (
    <div 
      className={`wrestling-card ${getTypeColor()} cursor-pointer hover:scale-105 transform transition-all duration-200`}
      onClick={onClick}
    >
      <div className="flex items-center gap-4">
        {/* Rank */}
        <div className="text-3xl font-black text-wrestling-red">
          #{wrestler.rank}
        </div>

        {/* Wrestler Image with Sentiment Frame */}
        <div className={`wrestler-image-container bg-wrestling-charcoal border-2 flex items-center justify-center ${
          wrestler.sentiment_score > 0.1 ? 'border-wrestling-green' : 
          wrestler.sentiment_score < -0.1 ? 'border-wrestling-red' : 
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
                  parent.innerHTML = `<div class="text-wrestling-gray text-xs text-center font-bold" style="width: 64px; height: 64px; display: flex; align-items: center; justify-content: center;">${initials}</div>`;
                }
              }}
              onLoad={(e) => {
                const target = e.target as HTMLImageElement;
                target.style.opacity = '1';
              }}
              style={{ 
                opacity: '0', 
                transition: 'opacity 0.3s ease'
              }}
            />
          ) : (
            <div className="text-wrestling-gray text-xs text-center font-bold" style={{width: '64px', height: '64px', display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
              {wrestler.name.split(' ').map(n => n[0]).join('').slice(0, 2)}
            </div>
          )}
        </div>

        {/* Wrestler Info */}
        <div className="flex-1">
          <h4 className="text-lg font-bold text-white mb-1">
            {wrestler.name.toUpperCase()}
          </h4>
          
          <div className="flex justify-between items-center">
            <span className="text-wrestling-gray text-sm">
              {wrestler.post_count} posts
            </span>
            <span className={`font-bold text-sm ${getScoreColor()}`}>
              {wrestler.sentiment_score > 0 ? '+' : ''}{wrestler.sentiment_score.toFixed(2)}
            </span>
          </div>
        </div>

        {/* Type Indicator */}
        <div className={`px-2 py-1 text-xs font-bold border ${
          type === 'loved' 
            ? 'border-wrestling-green text-wrestling-green' 
            : type === 'hated'
            ? 'border-wrestling-red text-wrestling-red'
            : 'border-wrestling-gray text-wrestling-gray'
        }`}>
          {type.toUpperCase()}
        </div>
      </div>
    </div>
  );
};

export default WrestlerCard;