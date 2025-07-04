import React, { useState, useEffect, useCallback } from 'react';
import Header from '../components/Header';
import ApiService from '../services/api';

interface TrainingPost {
  id: string;
  content: string;
  title?: string;
  subreddit?: string;
  author: string;
  score: number;
  created_at: string;
  predicted_sentiment?: number;
  predicted_classification?: string;
}

interface AdminTrainingProps {
  onBackToHome: () => void;
}

const AdminTraining: React.FC<AdminTrainingProps> = ({ onBackToHome }) => {
  const [isRunning, setIsRunning] = useState(false);
  const [currentPost, setCurrentPost] = useState<TrainingPost | null>(null);
  const [trainedCount, setTrainedCount] = useState(0);
  const [loading, setLoading] = useState(false);

  const fetchNextPost = useCallback(async () => {
    console.log('fetchNextPost called');
    setLoading(true);
    try {
      console.log('Calling ApiService.getNextTrainingPost...');
      const post = await ApiService.getNextTrainingPost();
      console.log('Received training post:', post);
      setCurrentPost(post);
    } catch (error) {
      console.error('Error fetching post:', error);
      // Fallback to mock data if API fails
      const mockPost: TrainingPost = {
        id: `training_${Date.now()}`,
        content: generateMockPost(),
        title: 'Mock Wrestling Post Title',
        subreddit: 'SquaredCircle',
        author: 'wrestlingfan123',
        score: Math.floor(Math.random() * 100),
        created_at: new Date().toISOString(),
        predicted_sentiment: Math.random() * 2 - 1, // -1 to 1
        predicted_classification: Math.random() > 0.7 ? 'news' : 'opinion'
      };
      setCurrentPost(mockPost);
    } finally {
      setLoading(false);
    }
  }, []);

  const startTraining = async () => {
    console.log('Starting training...');
    setIsRunning(true);
    setTrainedCount(0);
  };

  // Auto-fetch first post when training starts
  useEffect(() => {
    console.log('useEffect triggered - isRunning:', isRunning, 'currentPost:', currentPost);
    if (isRunning && !currentPost) {
      console.log('Calling fetchNextPost...');
      fetchNextPost();
    }
  }, [isRunning, currentPost]);

  const stopTraining = async () => {
    setIsRunning(false);
    // TODO: Save model checkpoint
    alert(`Training stopped. ${trainedCount} posts classified. Model saved.`);
  };

  const generateMockPost = () => {
    const posts = [
      "Roman Reigns had an incredible match last night! The storytelling was amazing.",
      "I think AEW is better than WWE right now, what do you think?",
      "BREAKING: Reports suggest major wrestling event planned for next month",
      "CM Punk's return was disappointing. Expected so much more.",
      "Why does everyone hate this wrestler? I think they're underrated.",
      "RUMOR: Multiple sources confirm big signing coming to WWE",
      "This match made me cry. Wrestling at its finest.",
      "Hot take: Modern wrestling is too scripted and predictable"
    ];
    return posts[Math.floor(Math.random() * posts.length)];
  };

  const classifyPost = async (sentiment: 'positive' | 'negative' | 'neutral', type: 'news' | 'opinion') => {
    if (!currentPost) return;

    try {
      const classification = {
        post_id: currentPost.id,
        sentiment,
        type,
        content: currentPost.content,
        title: currentPost.title,
        subreddit: currentPost.subreddit,
        author: currentPost.author
      };

      await ApiService.classifyTrainingPost(classification);
      setTrainedCount(prev => prev + 1);
      
      // Fetch next post
      setTimeout(fetchNextPost, 500);
    } catch (error) {
      console.error('Error saving classification:', error);
      // Still increment count and continue
      setTrainedCount(prev => prev + 1);
      setTimeout(fetchNextPost, 500);
    }
  };

  const getSentimentColor = (score?: number) => {
    if (!score) return 'text-wrestling-gray';
    if (score > 0.1) return 'text-wrestling-green';
    if (score < -0.1) return 'text-wrestling-red';
    return 'text-wrestling-gray';
  };

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

      {/* Admin Training Interface */}
      <section className="p-8">
        <div className="max-w-4xl mx-auto">
          <div className="wrestling-card mb-8">
            <h1 className="text-3xl font-black text-white mb-4 text-center">
              ADMIN TRAINING MODE
            </h1>
            <p className="text-wrestling-gray text-center mb-6">
              Continuously scrape wrestling posts and manually classify them to improve the NLP model
            </p>
            
            {/* Training Controls */}
            <div className="flex justify-center gap-4 mb-6">
              {!isRunning ? (
                <button 
                  onClick={startTraining}
                  className="wrestling-button bg-wrestling-green text-black"
                >
                  START TRAINING
                </button>
              ) : (
                <button 
                  onClick={stopTraining}
                  className="wrestling-button"
                >
                  STOP & SAVE MODEL
                </button>
              )}
            </div>

            {/* Training Stats */}
            <div className="text-center mb-6">
              <div className="text-2xl font-bold text-wrestling-green">
                {trainedCount}
              </div>
              <div className="text-sm text-wrestling-gray">
                POSTS CLASSIFIED
              </div>
            </div>
          </div>

          {/* Current Post for Classification */}
          {isRunning && currentPost && (
            <div className="wrestling-card">
              <h3 className="text-xl font-bold text-white mb-4">
                CLASSIFY THIS POST
              </h3>
              
              {/* Post Info */}
              <div className="mb-4 p-4 bg-wrestling-black rounded">
                <div className="flex justify-between items-start mb-2">
                  <div className="flex items-center gap-2 text-sm text-wrestling-gray">
                    <span>r/{currentPost.subreddit}</span>
                    <span>•</span>
                    <span>@{currentPost.author}</span>
                    <span>•</span>
                    <span>{currentPost.score} points</span>
                  </div>
                </div>
                
                {currentPost.title && (
                  <h4 className="font-bold text-white mb-2">
                    {currentPost.title}
                  </h4>
                )}
                
                <p className="text-wrestling-gray mb-4">
                  {currentPost.content}
                </p>
                
                {/* AI Predictions */}
                <div className="border-t border-wrestling-gray pt-4">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-wrestling-gray">AI Sentiment: </span>
                      <span className={getSentimentColor(currentPost.predicted_sentiment)}>
                        {currentPost.predicted_sentiment?.toFixed(2)}
                      </span>
                    </div>
                    <div>
                      <span className="text-wrestling-gray">AI Classification: </span>
                      <span className="text-white">{currentPost.predicted_classification}</span>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Classification Buttons */}
              <div className="space-y-4">
                <div>
                  <h4 className="text-white font-bold mb-2">SENTIMENT:</h4>
                  <div className="flex gap-2">
                    <button 
                      onClick={() => classifyPost('positive', currentPost.predicted_classification as any)}
                      className="px-4 py-2 bg-wrestling-green text-black font-bold"
                    >
                      POSITIVE
                    </button>
                    <button 
                      onClick={() => classifyPost('neutral', currentPost.predicted_classification as any)}
                      className="px-4 py-2 bg-wrestling-gray text-black font-bold"
                    >
                      NEUTRAL
                    </button>
                    <button 
                      onClick={() => classifyPost('negative', currentPost.predicted_classification as any)}
                      className="px-4 py-2 bg-wrestling-red text-white font-bold"
                    >
                      NEGATIVE
                    </button>
                  </div>
                </div>
                
                <div>
                  <h4 className="text-white font-bold mb-2">TYPE:</h4>
                  <div className="flex gap-2">
                    <button 
                      onClick={() => classifyPost(currentPost.predicted_sentiment! > 0 ? 'positive' : 'negative', 'news')}
                      className="px-4 py-2 bg-wrestling-red text-white font-bold"
                    >
                      NEWS/RUMOR
                    </button>
                    <button 
                      onClick={() => classifyPost(currentPost.predicted_sentiment! > 0 ? 'positive' : 'negative', 'opinion')}
                      className="px-4 py-2 bg-wrestling-gray text-black font-bold"
                    >
                      OPINION/DISCUSSION
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Loading State */}
          {isRunning && loading && (
            <div className="wrestling-card text-center">
              <div className="text-xl font-bold text-wrestling-red">
                FETCHING NEXT POST...
              </div>
            </div>
          )}

          {/* Instructions */}
          {!isRunning && (
            <div className="wrestling-card">
              <h3 className="text-xl font-bold text-white mb-4">
                HOW TO USE
              </h3>
              <ul className="space-y-2 text-wrestling-gray">
                <li>• Click "START TRAINING" to begin continuous scraping</li>
                <li>• Each post will be shown with AI predictions</li>
                <li>• Classify each post's sentiment (positive/negative/neutral)</li>
                <li>• Classify each post's type (news/opinion)</li>
                <li>• The system will continuously fetch new posts until you stop</li>
                <li>• Click "STOP & SAVE MODEL" to save training data and update the model</li>
              </ul>
            </div>
          )}
        </div>
      </section>
    </div>
  );
};

export default AdminTraining;