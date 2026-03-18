"""
country_identifier.py - License Plate Country Identification

Identifies the country of origin from a license plate using:
1. Visual detection of country codes (EU blue strip, etc.)
2. Regex pattern matching against known plate formats
3. Country code lookup table

Covers 60+ countries with confidence scoring.
"""

import re
import numpy as np
from PIL import Image


# ============================================================
# Country Code Lookup: International Vehicle Registration Codes
# ============================================================
COUNTRY_CODES = {
    # Europe
    'A': 'Austria', 'AL': 'Albania', 'AND': 'Andorra', 'AM': 'Armenia',
    'AZ': 'Azerbaijan', 'B': 'Belgium', 'BG': 'Bulgaria', 'BIH': 'Bosnia and Herzegovina',
    'BY': 'Belarus', 'CH': 'Switzerland', 'CY': 'Cyprus', 'CZ': 'Czech Republic',
    'D': 'Germany', 'DK': 'Denmark', 'E': 'Spain', 'EST': 'Estonia',
    'F': 'France', 'FIN': 'Finland', 'FL': 'Liechtenstein', 'GB': 'United Kingdom',
    'GE': 'Georgia', 'GR': 'Greece', 'H': 'Hungary', 'HR': 'Croatia',
    'I': 'Italy', 'IRL': 'Ireland', 'IS': 'Iceland', 'L': 'Luxembourg',
    'LT': 'Lithuania', 'LV': 'Latvia', 'M': 'Malta', 'MC': 'Monaco',
    'MD': 'Moldova', 'ME': 'Montenegro', 'MK': 'North Macedonia', 'MNE': 'Montenegro',
    'N': 'Norway', 'NL': 'Netherlands', 'P': 'Portugal', 'PL': 'Poland',
    'RO': 'Romania', 'RSM': 'San Marino', 'RUS': 'Russia', 'S': 'Sweden',
    'SK': 'Slovakia', 'SLO': 'Slovenia', 'SRB': 'Serbia', 'TR': 'Turkey',
    'UA': 'Ukraine', 'V': 'Vatican City',
    # Americas
    'USA': 'United States', 'CDN': 'Canada', 'MEX': 'Mexico',
    'BR': 'Brazil', 'RA': 'Argentina', 'RCH': 'Chile', 'CO': 'Colombia',
    'PE': 'Peru', 'EC': 'Ecuador', 'VE': 'Venezuela',
    # Asia & Middle East
    'CN': 'China', 'J': 'Japan', 'ROK': 'South Korea', 'IND': 'India',
    'PK': 'Pakistan', 'MY': 'Malaysia', 'RI': 'Indonesia', 'T': 'Thailand',
    'VN': 'Vietnam', 'RP': 'Philippines', 'IR': 'Iran', 'IL': 'Israel',
    'SA': 'Saudi Arabia', 'UAE': 'United Arab Emirates', 'KWT': 'Kuwait',
    'BRN': 'Bahrain', 'Q': 'Qatar', 'OM': 'Oman',
    # Africa
    'ZA': 'South Africa', 'ET': 'Egypt', 'MA': 'Morocco', 'TN': 'Tunisia',
    'DZ': 'Algeria', 'NG': 'Nigeria', 'EAK': 'Kenya', 'EAU': 'Uganda',
    # Oceania
    'AUS': 'Australia', 'NZ': 'New Zealand',
}


# ============================================================
# Plate Format Patterns: regex for common plate formats
# ============================================================
# Each entry: (compiled_regex, country, confidence_boost)
PLATE_PATTERNS = [
    # Turkey: 06 ABC 1234 or 34 AB 123
    (re.compile(r'^\d{2}\s?[A-Z]{1,3}\s?\d{2,4}$'), 'Turkey', 0.9),
    
    # Germany: XX-XX 1234 or X-XX 1234
    (re.compile(r'^[A-Z]{1,3}\s?-?\s?[A-Z]{1,2}\s?\d{1,4}$'), 'Germany', 0.7),
    
    # France: AA-123-AA
    (re.compile(r'^[A-Z]{2}\s?-?\s?\d{3}\s?-?\s?[A-Z]{2}$'), 'France', 0.9),
    
    # UK: AA12 ABC
    (re.compile(r'^[A-Z]{2}\d{2}\s?[A-Z]{3}$'), 'United Kingdom', 0.9),
    
    # Netherlands: XX-999-X or 99-XXX-9
    (re.compile(r'^[A-Z0-9]{2}\s?-?\s?[A-Z0-9]{3}\s?-?\s?[A-Z0-9]{1,2}$'), 'Netherlands', 0.6),
    
    # Italy: AA 123 AA
    (re.compile(r'^[A-Z]{2}\s?\d{3}\s?[A-Z]{2}$'), 'Italy', 0.8),
    
    # Spain: 1234 ABC
    (re.compile(r'^\d{4}\s?[A-Z]{3}$'), 'Spain', 0.9),
    
    # Poland: ABC 12345
    (re.compile(r'^[A-Z]{2,3}\s?\d{4,5}$'), 'Poland', 0.7),
    
    # Russia: A123AA 77
    (re.compile(r'^[A-Z]\d{3}[A-Z]{2}\s?\d{2,3}$'), 'Russia', 0.9),
    
    # USA: Various state formats, generally alphanumeric
    (re.compile(r'^[A-Z0-9]{1,3}\s?-?\s?[A-Z0-9]{3,4}$'), 'United States', 0.4),
    
    # Saudi Arabia: Arabic + digits
    (re.compile(r'^\d{1,4}\s?[A-Z]{1,3}$'), 'Saudi Arabia', 0.5),
    
    # UAE: Letter(s) + digits
    (re.compile(r'^[A-Z]{1,2}\s?\d{1,5}$'), 'United Arab Emirates', 0.5),
    
    # Brazil: ABC1D23 (Mercosur) or ABC-1234 (old)
    (re.compile(r'^[A-Z]{3}\s?-?\s?\d[A-Z]\d{2}$'), 'Brazil', 0.9),
    (re.compile(r'^[A-Z]{3}\s?-?\s?\d{4}$'), 'Brazil', 0.8),
    
    # Japan: Region + hiragana + 4 digits
    (re.compile(r'^\d{2,4}\s?[A-Z]\s?\d{2}\s?-?\s?\d{2}$'), 'Japan', 0.6),
    
    # South Korea: 12가 1234
    (re.compile(r'^\d{2,3}\s?[A-Z가-힣]\s?\d{4}$'), 'South Korea', 0.8),
    
    # India: KA01AB1234
    (re.compile(r'^[A-Z]{2}\s?\d{2}\s?[A-Z]{1,2}\s?\d{4}$'), 'India', 0.9),
    
    # Australia: various state formats
    (re.compile(r'^[A-Z]{3}\s?\d{3}$'), 'Australia', 0.5),
    (re.compile(r'^\d{3}\s?[A-Z]{3}$'), 'Australia', 0.5),
]


class CountryIdentifier:
    """
    Identify the country of origin from a license plate.
    
    Uses multiple methods:
    1. EU blue strip detection (visual)
    2. Country code extraction from plate text
    3. Plate format pattern matching
    
    Usage:
        identifier = CountryIdentifier()
        result = identifier.identify(plate_text='34 ABC 1234', plate_image=crop)
        print(result['country'], result['confidence'])
    """
    
    def identify(self, plate_text=None, plate_image=None):
        """
        Identify country from plate text and/or image.
        
        Args:
            plate_text: OCR-recognized plate text (cleaned)
            plate_image: PIL Image of the plate crop (for visual analysis)
            
        Returns:
            dict with:
                'country': str - identified country name
                'country_code': str - ISO-style code
                'confidence': float - confidence (0-1)
                'method': str - how the country was identified
                'all_matches': list of all candidate matches
        """
        candidates = []
        
        # Method 1: Check for EU blue strip (visual)
        if plate_image is not None:
            eu_result = self._detect_eu_strip(plate_image)
            if eu_result:
                candidates.append(eu_result)
        
        # Method 2: Country code in plate text
        if plate_text:
            code_result = self._check_country_code(plate_text)
            if code_result:
                candidates.append(code_result)
        
        # Method 3: Plate format pattern matching
        if plate_text:
            pattern_results = self._match_plate_format(plate_text)
            candidates.extend(pattern_results)
        
        # Select best candidate
        if candidates:
            # Sort by confidence (descending)
            candidates.sort(key=lambda x: x['confidence'], reverse=True)
            best = candidates[0]
            
            return {
                'country': best['country'],
                'country_code': best.get('country_code', ''),
                'confidence': round(best['confidence'], 4),
                'method': best['method'],
                'all_matches': candidates[:5]
            }
        
        return {
            'country': 'Unknown',
            'country_code': '',
            'confidence': 0.0,
            'method': 'none',
            'all_matches': []
        }
    
    def _detect_eu_strip(self, plate_image):
        """
        Detect the EU blue strip on the left side of the plate.
        
        EU plates have a blue vertical strip (~10% of width) on the left
        containing the EU flag and country code in white/yellow text.
        """
        try:
            import cv2
        except ImportError:
            return None
        
        img_np = np.array(plate_image)
        if len(img_np.shape) < 3:
            return None
        
        h, w = img_np.shape[:2]
        
        # Check left 12% of the plate
        strip_width = max(1, int(w * 0.12))
        strip = img_np[:, :strip_width]
        
        # Convert to HSV to detect blue
        hsv = cv2.cvtColor(strip, cv2.COLOR_RGB2HSV)
        
        # Blue range in HSV
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_ratio = np.sum(mask > 0) / mask.size
        
        if blue_ratio > 0.3:
            return {
                'country': 'European Union (member state)',
                'country_code': 'EU',
                'confidence': min(0.8, blue_ratio),
                'method': 'eu_blue_strip'
            }
        
        return None
    
    def _check_country_code(self, plate_text):
        """
        Check if the plate text starts with or contains a known country code.
        """
        text = plate_text.strip().upper()
        
        # Check longest codes first (3 letters, then 2, then 1)
        for length in [3, 2, 1]:
            prefix = text[:length]
            if prefix in COUNTRY_CODES:
                return {
                    'country': COUNTRY_CODES[prefix],
                    'country_code': prefix,
                    'confidence': 0.6 + (length * 0.1),  # Longer codes = more confident
                    'method': 'country_code_prefix'
                }
        
        return None
    
    def _match_plate_format(self, plate_text):
        """
        Match plate text against known country plate formats.
        """
        text = plate_text.strip().upper()
        matches = []
        
        for pattern, country, confidence in PLATE_PATTERNS:
            if pattern.match(text):
                # Find country code
                code = ''
                for c, name in COUNTRY_CODES.items():
                    if name == country:
                        code = c
                        break
                
                matches.append({
                    'country': country,
                    'country_code': code,
                    'confidence': confidence,
                    'method': 'format_pattern'
                })
        
        return matches


if __name__ == "__main__":
    print("CountryIdentifier - License Plate Country Detection")
    print("=" * 50)
    
    identifier = CountryIdentifier()
    
    # Test various plate formats
    test_plates = [
        ("34 ABC 1234", "Turkish plate"),
        ("AA-123-BB", "French plate"),
        ("AB12 CDE", "UK plate"),
        ("1234 ABC", "Spanish plate"),
        ("A123BC 77", "Russian plate"),
        ("KA01AB1234", "Indian plate"),
    ]
    
    for plate, desc in test_plates:
        result = identifier.identify(plate_text=plate)
        print(f"  {desc:20s} ({plate:15s}) → {result['country']:20s} "
              f"(conf: {result['confidence']:.2f}, method: {result['method']})")
