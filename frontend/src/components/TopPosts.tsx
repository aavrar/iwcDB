import React from 'react';
import { Post } from '../types';

interface TopPostsProps {
  positivePost: Post[];
  negativePosts: Post[];
}

const TopPosts: React.FC<TopPostsProps> = ({ positivePost, negativePosts }) => {
  const PostCard: React.FC<{ post: Post; type: 'positive' | 'negative' }> = ({ post, type }) => {
    const formatDate = (dateString: string) => {
      return new Date(dateString).toLocaleDateString();
    };

    const truncateText = (text: string, maxLength: number = 200) => {
      return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
    };

    return (
      <div className={`wrestling-card ${type === 'positive' ? 'border-wrestling-green' : 'border-wrestling-red'} mb-4`}>
        <div className="flex justify-between items-start mb-3">
          <div className="flex items-center gap-2">
            <span className={`px-2 py-1 text-xs font-bold ${
              type === 'positive' ? 'bg-wrestling-green text-black' : 'bg-wrestling-red text-white'
            }`}>
              {post.score > 0 ? `+${post.score}` : post.score}
            </span>
            <span className="text-wrestling-gray text-sm">@{post.author}</span>
            <span className="text-wrestling-gray text-sm">•</span>
            <span className="text-wrestling-gray text-sm">{formatDate(post.created_at)}</span>
          </div>
          {post.subreddit && (
            <span className="text-wrestling-gray text-xs">r/{post.subreddit}</span>
          )}
        </div>

        {post.title && (
          <h4 className="font-bold text-white mb-2 leading-tight">
            {truncateText(post.title, 100)}
          </h4>
        )}

        <p className="text-wrestling-gray leading-relaxed mb-3">
          {truncateText(post.content)}
        </p>

        {/* Media Preview */}
        {post.image_url && (
          <div className="mb-3">
            <img 
              src={post.image_url} 
              alt="Post media"
              className="max-w-full h-48 object-cover border border-wrestling-gray"
            />
          </div>
        )}

        {post.video_url && (
          <div className="mb-3">
            <video 
              src={post.video_url}
              controls
              className="max-w-full h-48 border border-wrestling-gray"
            />
          </div>
        )}

        <div className="flex justify-between items-center text-sm">
          <a 
            href={post.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-wrestling-red hover:text-white transition-colors duration-200"
          >
            VIEW POST →
          </a>
          {post.sentiment_score && (
            <span className={`px-2 py-1 text-xs font-bold ${
              post.sentiment_score > 0 ? 'text-wrestling-green' : 'text-wrestling-red'
            }`}>
              SENTIMENT: {post.sentiment_score > 0 ? '+' : ''}{post.sentiment_score.toFixed(2)}
            </span>
          )}
        </div>
      </div>
    );
  };

  return (
    <section className="bg-wrestling-black p-8">
      <div className="max-w-7xl mx-auto">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Positive Posts */}
          <div>
            <div className="flex items-center gap-3 mb-6">
              <div className="w-4 h-4 bg-wrestling-green"></div>
              <h3 className="text-2xl font-black text-white uppercase tracking-wider">
                TOP POSITIVE POSTS
              </h3>
            </div>
            {positivePost.length > 0 ? (
              positivePost.map((post) => (
                <PostCard key={post.id} post={post} type="positive" />
              ))
            ) : (
              <div className="wrestling-card text-center text-wrestling-gray py-8">
                <p>No positive posts found</p>
              </div>
            )}
          </div>

          {/* Negative Posts */}
          <div>
            <div className="flex items-center gap-3 mb-6">
              <div className="w-4 h-4 bg-wrestling-red"></div>
              <h3 className="text-2xl font-black text-white uppercase tracking-wider">
                TOP NEGATIVE POSTS
              </h3>
            </div>
            {negativePosts.length > 0 ? (
              negativePosts.map((post) => (
                <PostCard key={post.id} post={post} type="negative" />
              ))
            ) : (
              <div className="wrestling-card text-center text-wrestling-gray py-8">
                <p>No negative posts found</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
};

export default TopPosts;