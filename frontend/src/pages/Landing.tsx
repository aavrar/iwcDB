import React, { useState, useEffect } from 'react';
import Header from '../components/Header';
import SearchSection from '../components/SearchSection';
import WrestlerCard from '../components/WrestlerCard';
import NewsSection from '../components/NewsSection';
import TrendingSection from '../components/TrendingSection';
import { PopularWrestler } from '../types';
import ApiService from '../services/api';

interface LandingProps {
  onSearch: (query: string) => void;
  onCompare?: () => void;
  onWrestlerClick?: (wrestlerName: string) => void;
  onAdminTraining?: () => void;
  onLabeling?: () => void;
}

const Landing: React.FC<LandingProps> = ({ onSearch, onCompare, onWrestlerClick, onAdminTraining, onLabeling }) => {
  const [popularWrestlers, setPopularWrestlers] = useState<PopularWrestler[]>([]);
  const [lovedWrestlers, setLovedWrestlers] = useState<PopularWrestler[]>([]);
  const [hatedWrestlers, setHatedWrestlers] = useState<PopularWrestler[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadWrestlerData();
  }, []);

  const loadWrestlerData = async () => {
    try {
      setLoading(true);
      const [popular, loved, hated] = await Promise.all([
        ApiService.getPopularWrestlers(),
        ApiService.getMostLovedWrestlers(),
        ApiService.getMostHatedWrestlers()
      ]);
      
      setPopularWrestlers(popular.slice(0, 5));
      setLovedWrestlers(loved.slice(0, 5));
      setHatedWrestlers(hated.slice(0, 5));
    } catch (error) {
      console.error('Failed to load wrestler data:', error);
      // Mock data for development
      const mockWrestlers: PopularWrestler[] = [
        { name: 'CM Punk', post_count: 1250, sentiment_score: 0.75, rank: 1 },
        { name: 'Roman Reigns', post_count: 1180, sentiment_score: 0.35, rank: 2 },
        { name: 'Cody Rhodes', post_count: 980, sentiment_score: 0.68, rank: 3 },
        { name: 'Seth Rollins', post_count: 890, sentiment_score: 0.42, rank: 4 },
        { name: 'Jon Moxley', post_count: 720, sentiment_score: 0.58, rank: 5 },
      ];
      
      setPopularWrestlers(mockWrestlers);
      setLovedWrestlers(mockWrestlers.filter(w => w.sentiment_score > 0.5));
      setHatedWrestlers(mockWrestlers.map(w => ({...w, sentiment_score: -Math.abs(w.sentiment_score)})));
    } finally {
      setLoading(false);
    }
  };

  const handleWrestlerClick = (wrestlerName: string) => {
    if (onWrestlerClick) {
      onWrestlerClick(wrestlerName);
    } else {
      onSearch(wrestlerName);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-wrestling-black">
        <Header />
        <div className="flex items-center justify-center h-96">
          <div className="text-2xl font-bold text-wrestling-red">LOADING...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-wrestling-black">
      <Header 
        onCompare={onCompare} 
        showCompareButton={true}
        onAdminTraining={onAdminTraining}
        showAdminButton={true}
        onLabeling={onLabeling}
        showLabelingButton={true}
      />
      <SearchSection onSearch={onSearch} />
      
      {/* Jagged Divider */}
      <div className="h-8 bg-wrestling-red jagged-divider"></div>
      
      {/* Most Discussed Section */}
      <section className="bg-wrestling-charcoal p-8">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center gap-3 mb-6 justify-center">
            <div className="w-4 h-4 bg-wrestling-gray"></div>
            <h3 className="text-2xl font-black text-white uppercase tracking-wider">
              MOST DISCUSSED
            </h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
            {popularWrestlers.map((wrestler) => (
              <WrestlerCard
                key={wrestler.name}
                wrestler={wrestler}
                type="popular"
                onClick={() => handleWrestlerClick(wrestler.name)}
              />
            ))}
          </div>
        </div>
      </section>

      {/* Jagged Divider */}
      <div className="h-8 bg-wrestling-red jagged-divider"></div>

      {/* Most Loved/Hated Section */}
      <section className="bg-wrestling-charcoal p-8">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            
            {/* Most Loved */}
            <div>
              <div className="flex items-center gap-3 mb-6 justify-center">
                <div className="w-4 h-4 bg-wrestling-green"></div>
                <h3 className="text-2xl font-black text-white uppercase tracking-wider">
                  MOST LOVED
                </h3>
              </div>
              <div className="space-y-4">
                {lovedWrestlers.map((wrestler) => (
                  <WrestlerCard
                    key={wrestler.name}
                    wrestler={wrestler}
                    type="loved"
                    onClick={() => handleWrestlerClick(wrestler.name)}
                  />
                ))}
              </div>
            </div>

            {/* Most Hated */}
            <div>
              <div className="flex items-center gap-3 mb-6 justify-center">
                <div className="w-4 h-4 bg-wrestling-red"></div>
                <h3 className="text-2xl font-black text-white uppercase tracking-wider">
                  MOST HATED
                </h3>
              </div>
              <div className="space-y-4">
                {hatedWrestlers.map((wrestler) => (
                  <WrestlerCard
                    key={wrestler.name}
                    wrestler={wrestler}
                    type="hated"
                    onClick={() => handleWrestlerClick(wrestler.name)}
                  />
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>
      
      {/* Jagged Divider */}
      <div className="h-8 bg-wrestling-red jagged-divider"></div>
      
      {/* News Section */}
      <NewsSection />
      
      {/* Jagged Divider */}
      <div className="h-8 bg-wrestling-red jagged-divider"></div>
      
      {/* Trending Section */}
      <TrendingSection onWrestlerClick={handleWrestlerClick} />

      {/* Footer */}
      <footer className="bg-wrestling-black border-t-2 border-wrestling-red p-6">
        <div className="max-w-7xl mx-auto text-center">
          <p className="text-wrestling-gray text-sm">
            HEATMETER - REAL-TIME WRESTLING COMMUNITY SENTIMENT
          </p>
          <p className="text-wrestling-gray text-xs mt-2">
            Data sourced from Reddit wrestling communities
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Landing;