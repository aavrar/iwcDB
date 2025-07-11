# IWC Sentiment Tracker Backend

A comprehensive backend API for real-time sentiment analysis of the Internet Wrestling Community (IWC) on Twitter/X. Built with FastAPI, featuring modern web scraping, NLP sentiment analysis, and robust caching.

## Features

### = Multi-Strategy Web Scraping
- **Playwright**: Primary scraping method with JavaScript support and anti-detection
- **Twint**: Backup scraping method (no API required)
- **Requests**: Fallback method for basic scraping
- Automatic strategy switching and retry mechanisms

### >� Advanced NLP Sentiment Analysis
- Fine-tuned BERT models for wrestling-specific sentiment
- Wrestling context enhancement with domain-specific keywords
- Confidence scoring and uncertainty detection
- Batch processing for efficiency

### � High-Performance Caching
- Redis-based caching for tweets and sentiment data
- Multi-layer caching strategy (L1: Redis, L2: Database)
- Intelligent cache invalidation and TTL management
- Popular query tracking and analytics

### =� Security & Rate Limiting
- Comprehensive rate limiting with Redis backend
- Input sanitization and XSS protection
- CORS configuration and security headers
- Suspicious request detection and logging

### =� Real-time Analytics
- Sentiment timeline aggregation
- Top positive/negative tweets identification
- Query popularity tracking
- Comprehensive API statistics

## API Endpoints

### Search Endpoints
- `GET /api/v1/search` - Search and analyze tweets
- `GET /api/v1/search/history` - Get recent search queries
- `GET /api/v1/search/popular` - Get popular queries

### Sentiment Endpoints
- `GET /api/v1/sentiment/score` - Get aggregate sentiment score
- `GET /api/v1/sentiment/timeline` - Get sentiment timeline
- `GET /api/v1/sentiment/top-tweets` - Get top positive/negative tweets
- `GET /api/v1/sentiment/stats` - Get sentiment statistics

### System Endpoints
- `GET /` - API status and information
- `GET /health` - Detailed health check
- `GET /info` - API configuration information

### Admin Endpoints
- `GET /admin/cache/stats` - Cache statistics
- `DELETE /admin/cache/flush` - Flush all cache
- `DELETE /admin/cache/query/{query}` - Clear specific query cache

## Installation

### Prerequisites
- Python 3.11+
- Redis server
- Docker (optional)

### Local Development

1. **Clone the repository:**
```bash
git clone <repository-url>
cd IWCScraper/backend
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install Playwright browsers:**
```bash
playwright install chromium
```

5. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env with your configuration
```

6. **Start Redis server:**
```bash
redis-server
```

7. **Run the application:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Deployment

1. **Build and run with Docker Compose:**
```bash
docker-compose up --build
```

This will start:
- FastAPI application on port 8000
- Redis server on port 6379
- Celery worker for background tasks
- Flower monitoring on port 5555

## Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# App Configuration
DEBUG=false
SECRET_KEY=your-secret-key-here

# Database Configuration
DATABASE_URL=sqlite:///./iwc_sentiment.db
TURSO_DATABASE_URL=  # Optional: for Turso in production
TURSO_AUTH_TOKEN=    # Optional: for Turso in production

# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_PASSWORD=

# NLP Configuration
MODEL_NAME=cardiffnlp/twitter-roberta-base-sentiment-latest
MODEL_CACHE_DIR=./models
BATCH_SIZE=16

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000

# Scraping Configuration
SCRAPING_DELAY_MIN=1
SCRAPING_DELAY_MAX=3
MAX_TWEETS_PER_QUERY=100
PLAYWRIGHT_TIMEOUT=30000
```

### Database Options

#### SQLite (Development)
- Default: `sqlite:///./iwc_sentiment.db`
- No additional setup required
- Perfect for development and testing

#### Turso (Production)
- Set `TURSO_DATABASE_URL` and `TURSO_AUTH_TOKEN`
- Scalable SQLite-compatible database
- Free tier available

#### PostgreSQL (Alternative)
- Set `DATABASE_URL` to PostgreSQL connection string
- Requires PostgreSQL server setup

## API Usage Examples

### Search for tweets about a wrestler
```bash
curl "http://localhost:8000/api/v1/search?query=Roman%20Reigns&max_results=20"
```

### Get sentiment score for a query
```bash
curl "http://localhost:8000/api/v1/sentiment/score?query=Roman%20Reigns"
```

### Get sentiment timeline
```bash
curl "http://localhost:8000/api/v1/sentiment/timeline?query=Roman%20Reigns&hours=24"
```

### Get top tweets
```bash
curl "http://localhost:8000/api/v1/sentiment/top-tweets?query=Roman%20Reigns&count=5"
```

## Development

### Project Structure
```
backend/
   app/
      __init__.py
      main.py              # FastAPI application
      api/
         endpoints/
             search.py    # Search endpoints
             sentiment.py # Sentiment endpoints
      core/
         config.py        # Configuration management
         database.py      # Database connection
         scraper.py       # Web scraping service
         nlp.py          # NLP sentiment analysis
         cache.py        # Redis caching
         security.py     # Security and rate limiting
         logging.py      # Logging configuration
      models/
          tweet.py        # Database and response models
   requirements.txt
   Dockerfile
   docker-compose.yml
   .env.example
```

### Running Tests
```bash
pytest tests/ -v
```

### Code Quality
```bash
# Format code
black app/

# Sort imports
isort app/

# Lint code
flake8 app/

# Type checking
mypy app/
```

## Performance & Scaling

### Caching Strategy
- **L1 Cache**: Redis for frequently accessed data
- **L2 Cache**: Database for persistent storage
- **TTL Management**: Intelligent expiration based on data type

### Rate Limiting
- Per-IP rate limiting with Redis backend
- Endpoint-specific limits
- Graceful degradation under load

### Monitoring
- Health check endpoints
- Performance metrics
- Error tracking and logging

## Security

### Input Validation
- Query sanitization to prevent injection attacks
- Request size limits
- Content type validation

### Rate Limiting
- Per-IP request limits
- Endpoint-specific restrictions
- Suspicious request detection

### Security Headers
- XSS protection
- Content type options
- Frame options
- HSTS (in production)

## Troubleshooting

### Common Issues

1. **Playwright browser not found:**
```bash
playwright install chromium
```

2. **Redis connection failed:**
- Ensure Redis server is running
- Check Redis URL in environment variables

3. **Model download issues:**
- Ensure internet connection for initial model download
- Check model cache directory permissions

4. **Rate limiting errors:**
- Reduce request frequency
- Check rate limit configuration

### Logging
Logs are written to both console and file (`app.log`). Adjust log level in environment variables:
```bash
LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- HuggingFace Transformers for NLP models
- Playwright for modern web scraping
- FastAPI for the excellent web framework
- Redis for high-performance caching