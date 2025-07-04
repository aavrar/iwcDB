import asyncio
import random
import json
from typing import List, Dict, Optional
from datetime import datetime
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
import aiohttp
from app.core.config import settings
from app.core.logging import logger
from urllib.parse import quote
import time


class EnhancedTwitterScraper:
    """Enhanced Twitter scraper with anti-detection measures."""
    
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        
        # Realistic user agents
        self.user_agents = [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0"
        ]
        
        # Realistic screen resolutions
        self.viewports = [
            {'width': 1920, 'height': 1080},
            {'width': 1366, 'height': 768},
            {'width': 1536, 'height': 864},
            {'width': 1440, 'height': 900},
            {'width': 1280, 'height': 720}
        ]
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.init_browser()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_browser()
    
    async def init_browser(self):
        """Initialize Playwright browser with anti-detection."""
        try:
            self.playwright = await async_playwright().start()
            
            # Launch browser with stealth settings
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-accelerated-2d-canvas',
                    '--no-first-run',
                    '--no-zygote',
                    '--disable-gpu',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor',
                    '--disable-extensions',
                    '--disable-plugins',
                    '--disable-images',  # Speed up loading
                    '--disable-javascript-harmony-shipping',
                    '--disable-ipc-flooding-protection'
                ]
            )
            
            # Create context with random fingerprint
            viewport = random.choice(self.viewports)
            user_agent = random.choice(self.user_agents)
            
            self.context = await self.browser.new_context(
                user_agent=user_agent,
                viewport=viewport,
                locale='en-US',
                timezone_id='America/New_York',
                permissions=['geolocation'],
                extra_http_headers={
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Cache-Control': 'no-cache',
                    'Pragma': 'no-cache',
                    'Sec-Fetch-Dest': 'document',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-Site': 'none',
                    'Upgrade-Insecure-Requests': '1'
                }
            )
            
            # Add stealth scripts
            await self.context.add_init_script("""
                // Overwrite the `plugins` property to use a custom getter
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5]
                });
                
                // Overwrite the `webdriver` property to remove it
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
                
                // Mock languages
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en']
                });
                
                // Mock platform
                Object.defineProperty(navigator, 'platform', {
                    get: () => 'MacIntel'
                });
            """)
            
            logger.info("Enhanced browser initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced browser: {e}")
            raise
    
    async def close_browser(self):
        """Close browser."""
        try:
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if hasattr(self, 'playwright'):
                await self.playwright.stop()
            logger.info("Enhanced browser closed successfully")
        except Exception as e:
            logger.error(f"Error closing enhanced browser: {e}")
    
    async def scrape_with_guest_session(self, query: str, max_results: int = 50) -> List[Dict]:
        """Try to scrape using guest session (no login required)."""
        try:
            if not self.context:
                await self.init_browser()
            
            page = await self.context.new_page()
            
            # Block unnecessary resources
            await page.route("**/*.{png,jpg,jpeg,gif,svg,woff,woff2,ttf,css}", lambda route: route.abort())
            
            # Go to Twitter search without logging in
            search_url = f"https://twitter.com/search?q={quote(query)}&src=typed_query&f=live"
            
            # Add random delay
            await asyncio.sleep(random.uniform(2, 5))
            
            await page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
            
            # Wait for content or login prompt
            try:
                await page.wait_for_selector('article[data-testid="tweet"], div[data-testid="primaryColumn"]', timeout=10000)
            except:
                logger.warning("No tweet content found - might need authentication")
                await page.close()
                return []
            
            # Check if we hit a login wall
            login_elements = await page.query_selector_all('div[data-testid="loginButton"], a[href="/login"]')
            if login_elements:
                logger.warning("Hit login wall - Twitter requires authentication")
                await page.close()
                return []
            
            tweets = []
            scroll_attempts = 0
            max_scrolls = 5
            
            while len(tweets) < max_results and scroll_attempts < max_scrolls:
                # Extract tweets from current view
                tweet_elements = await page.query_selector_all('article[data-testid="tweet"]')
                
                for tweet_element in tweet_elements:
                    try:
                        tweet_data = await self._extract_tweet_data_enhanced(tweet_element)
                        if tweet_data and tweet_data['id'] not in [t['id'] for t in tweets]:
                            tweets.append(tweet_data)
                            
                            if len(tweets) >= max_results:
                                break
                    except Exception as e:
                        logger.warning(f"Error extracting tweet: {e}")
                        continue
                
                # Scroll down
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(random.uniform(1, 3))
                scroll_attempts += 1
            
            await page.close()
            logger.info(f"Enhanced scraping completed: {len(tweets)} tweets")
            return tweets
            
        except Exception as e:
            logger.error(f"Enhanced scraping failed: {e}")
            return []
    
    async def _extract_tweet_data_enhanced(self, tweet_element) -> Optional[Dict]:
        """Enhanced tweet data extraction."""
        try:
            # Extract text content
            text_element = await tweet_element.query_selector('[data-testid="tweetText"]')
            content = await text_element.inner_text() if text_element else ""
            
            # Extract username
            username_element = await tweet_element.query_selector('[data-testid="User-Name"] a')
            username_href = await username_element.get_attribute('href') if username_element else ""
            username = username_href.split('/')[-1] if username_href else "unknown"
            
            # Extract timestamp
            time_element = await tweet_element.query_selector('time')
            timestamp = await time_element.get_attribute('datetime') if time_element else None
            
            # Generate unique ID
            tweet_id = f"{username}_{hash(content)}_{int(time.time())}"
            
            return {
                'id': tweet_id,
                'content': content,
                'username': username,
                'datetime': timestamp or datetime.utcnow().isoformat(),
                'url': f"https://twitter.com/{username}/status/{tweet_id}",
                'source': 'enhanced_scraper'
            }
            
        except Exception as e:
            logger.warning(f"Enhanced extraction failed: {e}")
            return None
    
    async def scrape_with_mobile_user_agent(self, query: str, max_results: int = 50) -> List[Dict]:
        """Try scraping with mobile user agent (sometimes less restricted)."""
        try:
            # Create mobile context
            mobile_context = await self.browser.new_context(
                user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
                viewport={'width': 375, 'height': 667},
                device_scale_factor=2,
                is_mobile=True,
                has_touch=True
            )
            
            page = await mobile_context.new_page()
            
            # Mobile Twitter URL
            search_url = f"https://mobile.twitter.com/search?q={quote(query)}&src=typed_query"
            
            await page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(random.uniform(2, 4))
            
            # Mobile extraction logic would go here
            # This is a simplified version
            tweets = []
            
            await page.close()
            await mobile_context.close()
            
            logger.info(f"Mobile scraping completed: {len(tweets)} tweets")
            return tweets
            
        except Exception as e:
            logger.error(f"Mobile scraping failed: {e}")
            return []


# Usage costs:
# Playwright: FREE
# Proxy services: $10-50/month for residential proxies
# CAPTCHA solving: $1-3 per 1000 CAPTCHAs