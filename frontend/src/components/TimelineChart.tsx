import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { TimelinePoint } from '../types';

interface TimelineChartProps {
  data: TimelinePoint[];
  wrestlerName: string;
}

const TimelineChart: React.FC<TimelineChartProps> = ({ data, wrestlerName }) => {
  console.log('TimelineChart received data:', data);
  console.log('Data length:', data?.length);
  console.log('Wrestler name:', wrestlerName);
  
  // Transform data for chart display
  const chartData = data.map(point => ({
    date: new Date(point.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    sentiment: point.sentiment,
    posts: point.posts_count
  }));
  
  console.log('Transformed chart data:', chartData);

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-wrestling-charcoal border-2 border-wrestling-red p-3 text-white">
          <p className="font-bold text-wrestling-red">{label}</p>
          <p className="text-sm">
            Sentiment: <span className={`font-bold ${
              data.sentiment > 0.1 ? 'text-wrestling-green' : 
              data.sentiment < -0.1 ? 'text-wrestling-red' : 
              'text-wrestling-gray'
            }`}>
              {data.sentiment > 0 ? '+' : ''}{data.sentiment.toFixed(2)}
            </span>
          </p>
          <p className="text-sm text-wrestling-gray">
            Posts: {data.posts}
          </p>
        </div>
      );
    }
    return null;
  };

  if (!data || data.length === 0) {
    return (
      <div className="wrestling-card">
        <h3 className="text-xl font-bold text-white mb-4 text-center">
          SENTIMENT TIMELINE
        </h3>
        <div className="text-center text-wrestling-gray py-8">
          <p>No timeline data available for {wrestlerName}</p>
          <p className="text-sm mt-2">Data will appear as more posts are analyzed over time</p>
        </div>
      </div>
    );
  }

  return (
    <div className="wrestling-card">
      <h3 className="text-xl font-bold text-white mb-4 text-center">
        SENTIMENT TIMELINE - {wrestlerName.toUpperCase()}
      </h3>
      
      <div style={{ width: '100%', height: '300px', backgroundColor: '#0d0d0d', padding: '10px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart 
            data={chartData} 
            margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis 
              dataKey="date" 
              stroke="#9ca3af"
              fontSize={12}
              fontWeight="bold"
              tick={{ fill: '#9ca3af' }}
            />
            <YAxis 
              stroke="#9ca3af"
              fontSize={12}
              fontWeight="bold"
              domain={[-1, 1]}
              tickFormatter={(value) => value.toFixed(1)}
              tick={{ fill: '#9ca3af' }}
            />
            <Tooltip 
              content={<CustomTooltip />}
              cursor={{ stroke: '#ef4444', strokeWidth: 2 }}
            />
            <Line 
              type="monotone" 
              dataKey="sentiment" 
              stroke="#ef4444" 
              strokeWidth={3}
              dot={{ fill: '#ef4444', strokeWidth: 2, r: 4 }}
              activeDot={{ r: 6, stroke: '#ef4444', strokeWidth: 2, fill: '#ffffff' }}
              connectNulls={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      
      <div className="mt-4 text-center">
        <div className="flex justify-center gap-4 text-sm text-wrestling-gray">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-wrestling-green"></div>
            <span>Positive Sentiment</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-wrestling-red"></div>
            <span>Negative Sentiment</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TimelineChart;