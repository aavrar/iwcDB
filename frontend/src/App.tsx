import React, { useState } from 'react';
import Landing from './pages/Landing';
import SearchResults from './pages/SearchResults';
import Compare from './pages/Compare';
import WrestlerProfile from './pages/WrestlerProfile';
import AdminTraining from './pages/AdminTraining';
import LabelingInterface from './pages/LabelingInterface';

type AppState = 'landing' | 'search' | 'compare' | 'profile' | 'admin' | 'labeling';

function App() {
  const [currentState, setCurrentState] = useState<AppState>('landing');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [selectedWrestler, setSelectedWrestler] = useState<string>('');

  const handleSearch = (query: string) => {
    setSearchQuery(query);
    setCurrentState('search');
  };

  const handleBackToHome = () => {
    setCurrentState('landing');
    setSearchQuery('');
  };

  const handleCompare = () => {
    setCurrentState('compare');
  };

  const handleWrestlerProfile = (wrestlerName: string) => {
    setSelectedWrestler(wrestlerName);
    setCurrentState('profile');
  };

  const handleAdminTraining = () => {
    setCurrentState('admin');
  };

  const handleLabeling = () => {
    setCurrentState('labeling');
  };

  return (
    <div className="App">
      {currentState === 'landing' && (
        <Landing 
          onSearch={handleSearch} 
          onCompare={handleCompare}
          onWrestlerClick={handleWrestlerProfile}
          onAdminTraining={handleAdminTraining}
          onLabeling={handleLabeling}
        />
      )}
      {currentState === 'search' && (
        <SearchResults 
          query={searchQuery}
          onSearch={handleSearch}
          onBackToHome={handleBackToHome}
        />
      )}
      {currentState === 'compare' && (
        <Compare onBackToHome={handleBackToHome} />
      )}
      {currentState === 'profile' && (
        <WrestlerProfile 
          wrestlerName={selectedWrestler}
          onBackToHome={handleBackToHome}
          onSearch={handleSearch}
        />
      )}
      {currentState === 'admin' && (
        <AdminTraining 
          onBackToHome={handleBackToHome}
        />
      )}
      {currentState === 'labeling' && (
        <LabelingInterface />
      )}
    </div>
  );
}

export default App;