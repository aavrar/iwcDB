import React, { useState, useEffect, useRef, useCallback } from 'react';
import ApiService from '../services/api';

interface SearchSectionProps {
  onSearch: (query: string) => void;
  isLoading?: boolean;
}

interface WrestlerSuggestion {
  name: string;
  post_count: number;
  sentiment_score: number;
  image_url?: string;
}

const SearchSection: React.FC<SearchSectionProps> = ({ onSearch, isLoading = false }) => {
  const [query, setQuery] = useState('');
  const [suggestions, setSuggestions] = useState<WrestlerSuggestion[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const [loadingSuggestions, setLoadingSuggestions] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const suggestionsRef = useRef<HTMLDivElement>(null);

  // Debounced search for suggestions
  const debouncedSearch = useCallback((searchQuery: string) => {
    const timeoutId = setTimeout(async () => {
      if (searchQuery.length >= 2) {
        setLoadingSuggestions(true);
        try {
          const results = await ApiService.getWrestlerSuggestions(searchQuery, 6);
          setSuggestions(results);
          setShowSuggestions(true);
        } catch (error) {
          console.error('Error fetching suggestions:', error);
          setSuggestions([]);
        } finally {
          setLoadingSuggestions(false);
        }
      } else {
        setSuggestions([]);
        setShowSuggestions(false);
      }
    }, 300);
    
    return () => clearTimeout(timeoutId);
  }, []);

  useEffect(() => {
    debouncedSearch(query);
  }, [query, debouncedSearch]);

  // Close suggestions when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        suggestionsRef.current &&
        !suggestionsRef.current.contains(event.target as Node) &&
        inputRef.current &&
        !inputRef.current.contains(event.target as Node)
      ) {
        setShowSuggestions(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      setShowSuggestions(false);
      onSearch(query.trim());
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setQuery(value);
    setSelectedIndex(-1);
  };

  const handleSuggestionClick = (suggestion: WrestlerSuggestion) => {
    setQuery(suggestion.name);
    setShowSuggestions(false);
    onSearch(suggestion.name);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!showSuggestions || suggestions.length === 0) return;

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setSelectedIndex(prev => 
          prev < suggestions.length - 1 ? prev + 1 : prev
        );
        break;
      case 'ArrowUp':
        e.preventDefault();
        setSelectedIndex(prev => prev > 0 ? prev - 1 : -1);
        break;
      case 'Enter':
        if (selectedIndex >= 0) {
          e.preventDefault();
          handleSuggestionClick(suggestions[selectedIndex]);
        }
        break;
      case 'Escape':
        setShowSuggestions(false);
        setSelectedIndex(-1);
        break;
    }
  };

  const getSentimentColor = (score: number) => {
    if (score > 0.1) return 'text-wrestling-green';
    if (score < -0.1) return 'text-wrestling-red';
    return 'text-wrestling-gray';
  };

  return (
    <section className="bg-wrestling-black p-8">
      <div className="max-w-4xl mx-auto">
        <form onSubmit={handleSubmit} className="flex flex-col md:flex-row gap-4 relative">
          <div className="flex-1 relative">
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              onFocus={() => query.length >= 2 && setSuggestions.length > 0 && setShowSuggestions(true)}
              placeholder="Search wrestlers, events, or brands..."
              className="wrestling-input w-full text-xl"
              disabled={isLoading}
              autoComplete="off"
            />
            
            {/* Suggestions Dropdown */}
            {showSuggestions && (suggestions.length > 0 || loadingSuggestions) && (
              <div
                ref={suggestionsRef}
                className="absolute top-full left-0 right-0 z-50 bg-wrestling-charcoal border-2 border-wrestling-gray rounded-b-lg shadow-lg max-h-96 overflow-y-auto"
              >
                {loadingSuggestions ? (
                  <div className="p-4 text-center text-wrestling-gray">
                    <div className="text-sm font-bold">SEARCHING...</div>
                  </div>
                ) : (
                  suggestions.map((suggestion, index) => (
                    <div
                      key={suggestion.name}
                      onClick={() => handleSuggestionClick(suggestion)}
                      className={`p-3 cursor-pointer transition-colors duration-200 border-b border-wrestling-gray last:border-b-0 ${
                        index === selectedIndex
                          ? 'bg-wrestling-red bg-opacity-20'
                          : 'hover:bg-wrestling-gray hover:bg-opacity-10'
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        {/* Wrestler Image */}
                        <div className="w-10 h-10 bg-wrestling-black border border-wrestling-gray rounded overflow-hidden flex-shrink-0">
                          {suggestion.image_url ? (
                            <img
                              src={suggestion.image_url}
                              alt={suggestion.name}
                              className="w-full h-full object-cover"
                              onError={(e) => {
                                const target = e.target as HTMLImageElement;
                                target.style.display = 'none';
                                const parent = target.parentElement;
                                if (parent) {
                                  const initials = suggestion.name.split(' ').map(n => n[0]).join('').slice(0, 2);
                                  parent.innerHTML = `<div class="text-wrestling-red text-xs font-bold w-full h-full flex items-center justify-center">${initials}</div>`;
                                }
                              }}
                            />
                          ) : (
                            <div className="text-wrestling-red text-xs font-bold w-full h-full flex items-center justify-center">
                              {suggestion.name.split(' ').map(n => n[0]).join('').slice(0, 2)}
                            </div>
                          )}
                        </div>
                        
                        {/* Wrestler Info */}
                        <div className="flex-1">
                          <div className="text-white font-bold text-sm">
                            {suggestion.name}
                          </div>
                          <div className="flex items-center gap-3 text-xs">
                            <span className="text-wrestling-gray">
                              {suggestion.post_count} posts
                            </span>
                            <span className={`font-bold ${getSentimentColor(suggestion.sentiment_score)}`}>
                              {suggestion.sentiment_score > 0 ? '+' : ''}{suggestion.sentiment_score.toFixed(2)}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            )}
          </div>
          
          <button
            type="submit"
            disabled={isLoading || !query.trim()}
            className="wrestling-button text-xl px-8 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? 'ANALYZING...' : 'ANALYZE'}
          </button>
        </form>
        
        <div className="mt-4 flex flex-wrap gap-2">
          {['CM Punk', 'Roman Reigns', 'Cody Rhodes', 'WrestleMania', 'AEW Dynamite'].map((suggestion) => (
            <button
              key={suggestion}
              onClick={() => {
                setQuery(suggestion);
                onSearch(suggestion);
              }}
              className="suggestion-button"
              disabled={isLoading}
            >
              {suggestion}
            </button>
          ))}
        </div>
      </div>
    </section>
  );
};


export default SearchSection;