import asyncio
import aiohttp
import re
from typing import Optional, Dict, List
from urllib.parse import quote, unquote
from bs4 import BeautifulSoup
from app.core.logging import logger


class WikipediaImageScraper:
    """Scrape wrestler images from Wikipedia and other reliable sources."""
    
    def __init__(self):
        self.session = None
        self.headers = {
            'User-Agent': 'IWC-Pulse/1.0 (https://github.com/iwc-pulse; contact@iwc-pulse.com) Wrestling Sentiment Tracker'
        }
    
    async def _get_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=15)
            )
        return self.session
    
    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None
    
    async def search_wikipedia_image(self, wrestler_name: str) -> Optional[str]:
        """Search for wrestler image on Wikipedia."""
        try:
            session = await self._get_session()
            
            # Clean wrestler name for Wikipedia search
            clean_name = self._clean_wrestler_name(wrestler_name)
            search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(clean_name)}"
            
            async with session.get(search_url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Check if we have a thumbnail
                    if 'thumbnail' in data and 'source' in data['thumbnail']:
                        image_url = data['thumbnail']['source']
                        
                        # Validate the thumbnail isn't a vintage/wrong image
                        if self._is_valid_wrestler_image(image_url):
                            # Get higher resolution version
                            if '/thumb/' in image_url:
                                # Remove thumbnail sizing to get original
                                image_url = re.sub(r'/thumb/(.+)/\d+px-.+', r'/\1', image_url)
                            # Resize to consistent size
                            image_url = self._resize_wikipedia_image(image_url, 200)
                            return image_url
                    
                    # If no valid thumbnail in summary, try to get the page content
                    if 'content_urls' in data and 'desktop' in data['content_urls']:
                        page_url = data['content_urls']['desktop']['page']
                        return await self._extract_image_from_page(page_url)
                        
        except Exception as e:
            logger.debug(f"Wikipedia image search failed for {wrestler_name}: {e}")
        
        return None
    
    async def _extract_image_from_page(self, page_url: str) -> Optional[str]:
        """Extract the main image from a Wikipedia page."""
        try:
            session = await self._get_session()
            
            async with session.get(page_url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Look for the main infobox image (wrestler profile photo)
                    infobox = soup.find('table', {'class': 'infobox'})
                    if infobox:
                        # Find images within the infobox
                        imgs = infobox.find_all('img')
                        for img in imgs:
                            src = img.get('src')
                            if src and self._is_valid_wrestler_image(src):
                                if src.startswith('//'):
                                    src = 'https:' + src
                                # Ensure consistent sizing (200px width for better loading)
                                src = self._resize_wikipedia_image(src, 200)
                                return src
                    
                    # Fallback: look for portrait-style images in content
                    content_div = soup.find('div', {'id': 'mw-content-text'})
                    if content_div:
                        imgs = content_div.find_all('img')
                        for img in imgs:
                            src = img.get('src')
                            if src and self._is_valid_wrestler_image(src):
                                if src.startswith('//'):
                                    src = 'https:' + src
                                src = self._resize_wikipedia_image(src, 200)
                                return src
                        
        except Exception as e:
            logger.debug(f"Failed to extract image from page {page_url}: {e}")
        
        return None
    
    def _is_valid_wrestler_image(self, src: str) -> bool:
        """Check if image source is likely a wrestler profile photo."""
        if not src:
            return False
        
        # Skip common Wikipedia interface images and vintage photos
        skip_patterns = [
            'edit-icon', 'Commons-logo', 'Wikimedia_Commons',
            'Disambig', 'Question_book', 'ambox', 'wikimedia.org/static',
            'Nuvola', 'Crystal', 'Gnome', 'Information_icon',
            'Merge-', 'Split-', 'Lock-', 'Padlock',
            '193', '194', '195', '196', '197', '198',  # Vintage years
            '_1938_', '_1939_', '_1940_', '_1950_',   # Specific vintage years
            'black-and-white', 'vintage', 'historic',
            'Wrestling_-_Sikeston'  # Specific vintage wrestling photo
        ]
        
        for pattern in skip_patterns:
            if pattern in src:
                return False
        
        # Look for typical image file extensions
        if any(ext in src.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
            return True
        
        return False
    
    def _resize_wikipedia_image(self, src: str, width: int = 300) -> str:
        """Resize Wikipedia image to consistent width."""
        if '/thumb/' in src:
            # Extract the base image path and create new thumbnail URL
            match = re.search(r'/thumb/(.+?)/\d+px-(.+)', src)
            if match:
                base_path, filename = match.groups()
                return f"https://upload.wikimedia.org/wikipedia/commons/thumb/{base_path}/{width}px-{filename}"
        
        # If not a thumbnail URL, try to create one
        if 'upload.wikimedia.org' in src and '/commons/' in src:
            # Extract filename from commons URL
            parts = src.split('/commons/')
            if len(parts) == 2:
                path_parts = parts[1].split('/')
                if len(path_parts) >= 2:
                    hash_dir = path_parts[0]
                    sub_dir = path_parts[1] 
                    filename = '/'.join(path_parts[2:])
                    return f"https://upload.wikimedia.org/wikipedia/commons/thumb/{hash_dir}/{sub_dir}/{filename}/{width}px-{filename}"
        
        return src
    
    def _clean_wrestler_name(self, name: str) -> str:
        """Clean wrestler name for Wikipedia search."""
        # Common wrestling name patterns
        name = name.strip()
        
        # Handle common variations - try wrestler page first, then real name
        name_mappings = {
            'CM Punk': 'CM_Punk',
            'Jon Moxley': 'Jon_Moxley', 
            'Cody Rhodes': 'Cody_Rhodes',
            'Roman Reigns': 'Roman_Reigns',
            'Seth Rollins': 'Seth_Rollins', 
            'Rhea Ripley': 'Rhea_Ripley',
            'Bianca Belair': 'Bianca_Belair',
            'Gunther': 'Gunther_(wrestler)',
            'Drew McIntyre': 'Drew_McIntyre',
            'Damian Priest': 'Damian_Priest',
            'Liv Morgan': 'Liv_Morgan',
            'Dominik Mysterio': 'Dominik_Mysterio',
            'Jey Uso': 'Jey_Uso',
            'Jimmy Uso': 'Jimmy_Uso',
            'LA Knight': 'LA_Knight',
            'Kevin Owens': 'Kevin_Owens',
            'Randy Orton': 'Randy_Orton',
            'John Cena': 'John_Cena',
            'Brock Lesnar': 'Brock_Lesnar'
        }
        
        # Return mapped name if available, otherwise format for Wikipedia
        if name in name_mappings:
            return name_mappings[name]
        else:
            # Convert spaces to underscores for Wikipedia URLs
            return name.replace(' ', '_')


class CageMatchImageScraper:
    """Backup image scraper for CageMatch.net wrestling database."""
    
    def __init__(self):
        self.session = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    async def _get_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=10)
            )
        return self.session
    
    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None
    
    async def search_cagematch_image(self, wrestler_name: str) -> Optional[str]:
        """Search for wrestler image on CageMatch.net."""
        try:
            session = await self._get_session()
            
            # CageMatch search URL
            search_url = f"https://www.cagematch.net/?id=2&nr=&name={quote(wrestler_name)}"
            
            async with session.get(search_url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Look for wrestler profile image
                    img_elements = soup.find_all('img', {'src': True})
                    for img in img_elements:
                        src = img['src']
                        if 'wrestler' in src.lower() and src.endswith(('.jpg', '.jpeg', '.png')):
                            if not src.startswith('http'):
                                src = 'https://www.cagematch.net' + src
                            return src
                            
        except Exception as e:
            logger.debug(f"CageMatch image search failed for {wrestler_name}: {e}")
        
        return None


class MultiSourceImageScraper:
    """Main image scraper that tries multiple sources."""
    
    def __init__(self):
        self.wikipedia_scraper = WikipediaImageScraper()
        self.cagematch_scraper = CageMatchImageScraper()
    
    async def get_wrestler_image(self, wrestler_name: str) -> Optional[str]:
        """Get wrestler image from multiple sources with fallback."""
        logger.info(f"Searching for image: {wrestler_name}")
        
        # Try Wikipedia first (most reliable)
        try:
            image_url = await self.wikipedia_scraper.search_wikipedia_image(wrestler_name)
            if image_url and await self._validate_image_url(image_url):
                logger.info(f"Found Wikipedia image for {wrestler_name}")
                return image_url
        except Exception as e:
            logger.debug(f"Wikipedia search failed for {wrestler_name}: {e}")
        
        # Try CageMatch as backup
        try:
            image_url = await self.cagematch_scraper.search_cagematch_image(wrestler_name)
            if image_url and await self._validate_image_url(image_url):
                logger.info(f"Found CageMatch image for {wrestler_name}")
                return image_url
        except Exception as e:
            logger.debug(f"CageMatch search failed for {wrestler_name}: {e}")
        
        # Fallback to a wrestling-themed placeholder
        logger.info(f"No image found for {wrestler_name}, using placeholder")
        return self._generate_wrestling_placeholder(wrestler_name)
    
    async def _validate_image_url(self, url: str) -> bool:
        """Validate that an image URL is accessible and is actually an image."""
        try:
            session = await self.wikipedia_scraper._get_session()
            async with session.head(url) as response:
                if response.status == 200:
                    content_type = response.headers.get('content-type', '')
                    return 'image' in content_type.lower()
                return False
        except Exception:
            return False
    
    def _generate_wrestling_placeholder(self, wrestler_name: str) -> str:
        """Generate a wrestling-themed placeholder image."""
        # Use wrestler initials for better placeholder
        initials = ''.join([word[0].upper() for word in wrestler_name.split() if word])[:2]
        
        # Wrestling-themed colors
        colors = ['ef4444', '10b981', 'f59e0b', '8b5cf6', '06b6d4', 'ec4899']
        color = colors[hash(wrestler_name) % len(colors)]
        
        return f"https://via.placeholder.com/400x500/{color}/ffffff?text={initials}&font=bold"
    
    async def close(self):
        """Close all scraper sessions."""
        await self.wikipedia_scraper.close()
        await self.cagematch_scraper.close()


# Global instance
image_scraper = MultiSourceImageScraper()