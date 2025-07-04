import React, { useState, useEffect } from 'react';
import ApiService from '../services/api';

interface Post {
  post_id: string;
  content: string;
  username: string;
  subreddit: string;
  predicted_content_type: string;
  predicted_sentiment: string;
  content_confidence: number;
  sentiment_confidence: number;
  needs_review: boolean;
  manual_content_type?: string;
  manual_sentiment?: string;
}

interface LabelingStats {
  total: number;
  labeled: number;
  remaining: number;
  news: number;
  rumor: number;
  opinion: number;
}

const LabelingInterface: React.FC = () => {
  const [posts, setPosts] = useState<Post[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [labels, setLabels] = useState<{[key: string]: {contentType: string, sentiment: string}}>({});
  const [stats, setStats] = useState<LabelingStats>({
    total: 0, labeled: 0, remaining: 0, news: 0, rumor: 0, opinion: 0
  });

  const [csvFiles, setCsvFiles] = useState<any[]>([]);
  const [selectedFile, setSelectedFile] = useState<string>('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadAvailableFiles();
  }, []);

  const loadAvailableFiles = async () => {
    try {
      const response = await ApiService.makeRequest('/labeling/csv-files');
      setCsvFiles(response.csv_files || []);
      
      // Auto-select the most recent file
      if (response.csv_files && response.csv_files.length > 0) {
        setSelectedFile(response.csv_files[0].filename);
      }
    } catch (error) {
      console.error('Failed to load CSV files:', error);
    }
  };

  const loadPosts = async (filename: string) => {
    if (!filename) return;
    
    try {
      setLoading(true);
      const response = await ApiService.makeRequest(`/labeling/load/${filename}`);
      setPosts(response.posts || []);
      
      // Load existing labels
      const existingLabels: {[key: string]: {contentType: string, sentiment: string}} = {};
      response.posts.forEach((post: Post) => {
        if (post.manual_content_type) {
          existingLabels[post.post_id] = {
            contentType: post.manual_content_type,
            sentiment: post.manual_sentiment || ''
          };
        }
      });
      
      setLabels(existingLabels);
      updateStats(response.posts, existingLabels);
      setLoading(false);
    } catch (error) {
      console.error('Failed to load posts:', error);
      setLoading(false);
    }
  };

  useEffect(() => {
    if (selectedFile) {
      loadPosts(selectedFile);
    }
  }, [selectedFile]);

  const updateStats = (allPosts: Post[], currentLabels: {[key: string]: {contentType: string, sentiment: string}}) => {
    const labeled = Object.keys(currentLabels).length;
    const contentTypeCounts = { news: 0, rumor: 0, opinion: 0 };
    
    Object.values(currentLabels).forEach(label => {
      if (label.contentType in contentTypeCounts) {
        contentTypeCounts[label.contentType as keyof typeof contentTypeCounts]++;
      }
    });

    setStats({
      total: allPosts.length,
      labeled,
      remaining: allPosts.length - labeled,
      ...contentTypeCounts
    });
  };

  const handleLabel = (contentType: string, sentiment?: string) => {
    const currentPost = posts[currentIndex];
    if (!currentPost) return;

    const newLabels = {
      ...labels,
      [currentPost.post_id]: {
        contentType,
        sentiment: sentiment || ''
      }
    };

    setLabels(newLabels);
    updateStats(posts, newLabels);

    // Move to next post
    if (currentIndex < posts.length - 1) {
      setCurrentIndex(currentIndex + 1);
    }
  };

  const goToPrevious = () => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1);
    }
  };

  const goToNext = () => {
    if (currentIndex < posts.length - 1) {
      setCurrentIndex(currentIndex + 1);
    }
  };

  const saveProgress = async () => {
    if (!selectedFile) return;
    
    try {
      // Convert labels to API format
      const labelsForApi: {[key: string]: {content_type: string, sentiment: string, notes: string}} = {};
      Object.entries(labels).forEach(([postId, label]) => {
        labelsForApi[postId] = {
          content_type: label.contentType,
          sentiment: label.sentiment,
          notes: ''
        };
      });

      const response = await ApiService.makeRequest(`/labeling/save/${selectedFile}`, {
        method: 'POST',
        body: JSON.stringify(labelsForApi)
      });

      alert(`Saved! Progress: ${response.progress.labeled}/${response.progress.total} (${response.progress.percentage}%)`);
    } catch (error) {
      console.error('Failed to save progress:', error);
      alert('Failed to save progress');
    }
  };

  const exportLabels = async () => {
    if (!selectedFile) return;
    
    try {
      const response = await ApiService.makeRequest(`/labeling/export/${selectedFile}`, {
        method: 'POST'
      });
      
      alert(`Exported ${response.statistics.total_labeled} labeled posts to ${response.export_filename}`);
    } catch (error) {
      console.error('Failed to export:', error);
      alert('Failed to export labeled data');
    }
  };

  if (loading || posts.length === 0) {
    return (
      <div className="min-h-screen bg-wrestling-black">
        <div className="bg-wrestling-charcoal border-b-4 border-wrestling-red p-4">
          <div className="max-w-4xl mx-auto">
            <h1 className="text-3xl font-black uppercase tracking-wider mb-4">Wrestling Data Labeling</h1>
            
            {csvFiles.length > 0 ? (
              <div className="space-y-4">
                <div className="text-white">
                  <label className="block text-sm font-bold mb-2">Select CSV File:</label>
                  <select 
                    value={selectedFile}
                    onChange={(e) => setSelectedFile(e.target.value)}
                    className="bg-wrestling-black text-white p-2 rounded border border-wrestling-gray w-full max-w-md"
                  >
                    <option value="">-- Select a file --</option>
                    {csvFiles.map(file => (
                      <option key={file.filename} value={file.filename}>
                        {file.filename} ({file.post_count} posts, {file.progress}% labeled)
                      </option>
                    ))}
                  </select>
                </div>
                
                {loading && selectedFile && (
                  <div className="text-white text-xl">Loading posts from {selectedFile}...</div>
                )}
              </div>
            ) : (
              <div className="text-white text-xl">No CSV files found. Generate training data first.</div>
            )}
          </div>
        </div>
      </div>
    );
  }

  const currentPost = posts[currentIndex];
  const currentLabel = labels[currentPost?.post_id];

  return (
    <div className="min-h-screen bg-wrestling-black text-white">
      {/* Header with stats */}
      <div className="bg-wrestling-charcoal border-b-4 border-wrestling-red p-4">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-black uppercase tracking-wider mb-4">Wrestling Data Labeling</h1>
          
          <div className="grid grid-cols-2 md:grid-cols-6 gap-4 text-sm">
            <div className="bg-wrestling-black p-3 rounded">
              <div className="text-wrestling-gray">Progress</div>
              <div className="text-xl font-bold">{stats.labeled}/{stats.total}</div>
            </div>
            <div className="bg-wrestling-black p-3 rounded">
              <div className="text-wrestling-gray">Remaining</div>
              <div className="text-xl font-bold text-wrestling-red">{stats.remaining}</div>
            </div>
            <div className="bg-wrestling-black p-3 rounded">
              <div className="text-wrestling-gray">News</div>
              <div className="text-xl font-bold text-blue-400">{stats.news}</div>
            </div>
            <div className="bg-wrestling-black p-3 rounded">
              <div className="text-wrestling-gray">Rumor</div>
              <div className="text-xl font-bold text-yellow-400">{stats.rumor}</div>
            </div>
            <div className="bg-wrestling-black p-3 rounded">
              <div className="text-wrestling-gray">Opinion</div>
              <div className="text-xl font-bold text-wrestling-green">{stats.opinion}</div>
            </div>
            <div className="bg-wrestling-black p-3 rounded flex gap-2">
              <button 
                onClick={saveProgress}
                className="wrestling-button flex-1 text-xs"
              >
                SAVE
              </button>
              <button 
                onClick={exportLabels}
                className="wrestling-button flex-1 text-xs"
              >
                EXPORT
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main labeling interface */}
      <div className="max-w-4xl mx-auto p-6">
        {/* Navigation */}
        <div className="flex justify-between items-center mb-6">
          <button 
            onClick={goToPrevious}
            disabled={currentIndex === 0}
            className="wrestling-button disabled:opacity-50"
          >
            ← PREVIOUS
          </button>
          
          <div className="text-center">
            <div className="text-2xl font-bold">Post {currentIndex + 1} of {posts.length}</div>
            {currentPost?.needs_review && (
              <div className="text-wrestling-red text-sm mt-1">⚠ NEEDS REVIEW</div>
            )}
          </div>
          
          <button 
            onClick={goToNext}
            disabled={currentIndex === posts.length - 1}
            className="wrestling-button disabled:opacity-50"
          >
            NEXT →
          </button>
        </div>

        {/* Post content */}
        <div className="bg-wrestling-charcoal p-6 rounded-lg mb-6">
          <div className="flex justify-between items-start mb-4">
            <div>
              <div className="text-wrestling-gray text-sm">
                r/{currentPost?.subreddit} • u/{currentPost?.username}
              </div>
            </div>
            <div className="text-right text-sm">
              <div className="text-wrestling-gray">AI Prediction:</div>
              <div className="text-blue-400">{currentPost?.predicted_content_type}</div>
              {currentPost?.predicted_sentiment && (
                <div className="text-green-400">{currentPost?.predicted_sentiment}</div>
              )}
              <div className="text-xs text-wrestling-gray mt-1">
                Confidence: {(currentPost?.content_confidence * 100).toFixed(0)}%
              </div>
            </div>
          </div>
          
          <div className="text-lg leading-relaxed">
            {currentPost?.content}
          </div>
        </div>

        {/* Current label display */}
        {currentLabel && (
          <div className="bg-wrestling-green/20 border border-wrestling-green p-4 rounded mb-6">
            <div className="text-wrestling-green font-bold">
              Labeled as: {currentLabel.contentType}
              {currentLabel.sentiment && ` (${currentLabel.sentiment})`}
            </div>
          </div>
        )}

        {/* Labeling buttons */}
        <div className="space-y-6">
          {/* Content Type */}
          <div>
            <h3 className="text-xl font-bold mb-3">Content Type:</h3>
            <div className="grid grid-cols-3 gap-4">
              <button 
                onClick={() => handleLabel('news')}
                className="wrestling-button bg-blue-600 hover:bg-blue-700 p-4 text-center"
              >
                <div className="font-bold text-lg">NEWS</div>
                <div className="text-sm opacity-80">Official announcements, confirmed info</div>
              </button>
              
              <button 
                onClick={() => handleLabel('rumor')}
                className="wrestling-button bg-yellow-600 hover:bg-yellow-700 p-4 text-center"
              >
                <div className="font-bold text-lg">RUMOR</div>
                <div className="text-sm opacity-80">Speculation, unconfirmed reports</div>
              </button>
              
              <button 
                onClick={() => handleLabel('opinion')}
                className="wrestling-button bg-wrestling-green hover:bg-green-700 p-4 text-center"
              >
                <div className="font-bold text-lg">OPINION</div>
                <div className="text-sm opacity-80">Fan reactions, personal takes</div>
              </button>
            </div>
          </div>

          {/* Sentiment (for opinions) */}
          {(currentLabel?.contentType === 'opinion' || currentPost?.predicted_content_type === 'opinion') && (
            <div>
              <h3 className="text-xl font-bold mb-3">Sentiment (for opinions):</h3>
              <div className="grid grid-cols-3 gap-4">
                <button 
                  onClick={() => handleLabel('opinion', 'negative')}
                  className="wrestling-button bg-red-600 hover:bg-red-700 p-3 text-center"
                >
                  NEGATIVE
                </button>
                
                <button 
                  onClick={() => handleLabel('opinion', 'neutral')}
                  className="wrestling-button bg-gray-600 hover:bg-gray-700 p-3 text-center"
                >
                  NEUTRAL
                </button>
                
                <button 
                  onClick={() => handleLabel('opinion', 'positive')}
                  className="wrestling-button bg-green-600 hover:bg-green-700 p-3 text-center"
                >
                  POSITIVE
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Keyboard shortcuts help */}
        <div className="mt-8 p-4 bg-wrestling-black/50 rounded text-sm text-wrestling-gray">
          <div className="font-bold mb-2">Keyboard Shortcuts:</div>
          <div className="grid grid-cols-3 gap-4">
            <div>N = News | R = Rumor | O = Opinion</div>
            <div>1 = Negative | 2 = Neutral | 3 = Positive</div>
            <div>← → = Navigate | Space = Next</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LabelingInterface;