"""
country_identifier.py - License Plate Country Identification

Identifies the country of origin from a license plate using:
1. Visual detection of country codes (EU blue strip, etc.)
2. Regex pattern matching against known plate formats
3. Country code lookup table

Covers Europe, Turkey, and Turkey's neighboring countries.
"""

import re
import numpy as np
from PIL import Image


# ============================================================
# Country Code Lookup: International Vehicle Registration Codes
# Scope: Europe + Turkey + Turkey's neighbors
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
    # Turkey's neighbors (non-European)
    'IR': 'Iran', 'IRQ': 'Iraq', 'SYR': 'Syria',
}


# ============================================================
# Plate Format Patterns: regex for common plate formats
# ============================================================
# Each entry: (compiled_regex, country, confidence_boost)
PLATE_PATTERNS = [
    # Turkey: 06 ABC 1234 or 34 AB 123
    (re.compile(r'^\d{2}\s?[A-Z]{1,3}\s?\d{2,4}$'), 'Turkey', 'TR', 0.9),
    
    # Germany: XX-XX 1234 or X-XX 1234
    (re.compile(r'^[A-Z]{1,3}\s?-?\s?[A-Z]{1,2}\s?\d{1,4}[EH]?$'), 'Germany', 'D', 0.7),
    
    # France: AA-123-AA
    (re.compile(r'^[A-Z]{2}\s?-?\s?\d{3}\s?-?\s?[A-Z]{2}$'), 'France', 'F', 0.9),
    
    # UK: AA12 ABC
    (re.compile(r'^[A-Z]{2}\d{2}\s?[A-Z]{3}$'), 'United Kingdom', 'GB', 0.9),
    
    # Netherlands: X-999-XX, XX-999-X, 99-XXX-9, 9-XXX-99, XX-XX-99 etc.
    (re.compile(r'^[A-Z]{1,2}\s?-\s?\d{3}\s?-\s?[A-Z]{1,2}$'), 'Netherlands', 'NL', 0.92),
    (re.compile(r'^\d{1,2}\s?-\s?[A-Z]{3}\s?-\s?\d{1,2}$'), 'Netherlands', 'NL', 0.92),
    (re.compile(r'^[A-Z]{2}\s?-\s?[A-Z]{2}\s?-\s?\d{2}$'), 'Netherlands', 'NL', 0.88),
    (re.compile(r'^\d{2}\s?-\s?[A-Z]{2}\s?-\s?[A-Z]{2}$'), 'Netherlands', 'NL', 0.88),
    (re.compile(r'^[A-Z]{2}\s?-\s?\d{2}\s?-\s?[A-Z]{2}$'), 'Netherlands', 'NL', 0.88),
    # Fallback: generic 3-segment sidecode with hyphens
    (re.compile(r'^[A-Z0-9]{1,3}\s?-\s?[A-Z0-9]{2,3}\s?-\s?[A-Z0-9]{1,3}$'), 'Netherlands', 'NL', 0.6),
    
    # Belgium: 1-ABC-234
    (re.compile(r'^\d\s?-?\s?[A-Z]{3}\s?-?\s?\d{3}$'), 'Belgium', 'B', 0.9),
    
    # Italy: AA 123 AA
    (re.compile(r'^[A-Z]{2}\s?\d{3}\s?[A-Z]{2}$'), 'Italy', 'I', 0.8),
    
    # Spain: 1234 ABC
    (re.compile(r'^\d{4}\s?[A-Z]{3}$'), 'Spain', 'E', 0.9),
    
    # Poland: ABC 12345 or ABC1234
    (re.compile(r'^[A-Z]{2,3}\s?\d{4,5}$'), 'Poland', 'PL', 0.7),
    (re.compile(r'^[A-Z]{2}\s?\d{5}$'), 'Poland', 'PL', 0.8),
    
    # Russia: A123AA 77 or A123AA 777
    (re.compile(r'^[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}\s?\d{2,3}$'), 'Russia', 'RUS', 0.95),
    (re.compile(r'^[A-Z]\d{3}[A-Z]{2}\s?\d{2,3}$'), 'Russia', 'RUS', 0.85),
    
    # Austria: W-12345A or AB-12345
    (re.compile(r'^[A-Z]{1,2}\s?-?\s?\d{3,5}\s?[A-Z]?$'), 'Austria', 'A', 0.5),
    
    # Sweden: ABC 123 or ABC 12A
    (re.compile(r'^[A-Z]{3}\s?\d{2}[A-Z0-9]$'), 'Sweden', 'S', 0.8),
    
    # Denmark: AB 12 345 or AB12345
    (re.compile(r'^[A-Z]{2}\s?\d{2}\s?\d{3}$'), 'Denmark', 'DK', 0.7),
    
    # Norway: AB 12345
    (re.compile(r'^[A-Z]{2}\s?\d{5}$'), 'Norway', 'N', 0.8),
    
    # Portugal: AA-12-AA or 12-AA-12 or AA-12-12
    (re.compile(r'^[A-Z]{2}\s?-?\s?\d{2}\s?-?\s?[A-Z]{2}$'), 'Portugal', 'P', 0.7),
    (re.compile(r'^\d{2}\s?-?\s?[A-Z]{2}\s?-?\s?\d{2}$'), 'Portugal', 'P', 0.7),
    
    # Czech Republic: 1A2 3456
    (re.compile(r'^\d[A-Z]\d\s?\d{4}$'), 'Czech Republic', 'CZ', 0.85),
    
    # Romania: B-123-ABC or AB-12-ABC
    (re.compile(r'^[A-Z]{1,2}\s?-?\s?\d{2,3}\s?-?\s?[A-Z]{3}$'), 'Romania', 'RO', 0.7),
    
    # Greece: AAA-1234
    (re.compile(r'^[A-Z]{3}\s?-?\s?\d{4}$'), 'Greece', 'GR', 0.7),
    
    # Switzerland: AG 123456 or ZH 1234
    (re.compile(r'^[A-Z]{2}\s?\d{4,6}$'), 'Switzerland', 'CH', 0.6),
    
    # Croatia: ZG-1234-AA
    (re.compile(r'^[A-Z]{2}\s?-?\s?\d{3,4}\s?-?\s?[A-Z]{1,2}$'), 'Croatia', 'HR', 0.7),
    
    # Hungary: AAA-123
    (re.compile(r'^[A-Z]{3}\s?-?\s?\d{3}$'), 'Hungary', 'H', 0.6),
    
    # Bulgaria: A 1234 AB
    (re.compile(r'^[A-Z]{1,2}\s?\d{4}\s?[A-Z]{2}$'), 'Bulgaria', 'BG', 0.8),
    
    # Finland: ABC-123
    (re.compile(r'^[A-Z]{3}\s?-\s?\d{3}$'), 'Finland', 'FIN', 0.85),
    
    # --- Turkey's neighbors ---
    
    # Georgia: AA-123-AAA or AA-1234 (region code + digits + letters)
    (re.compile(r'^[A-Z]{2}\s?-?\s?\d{3}\s?-?\s?[A-Z]{3}$'), 'Georgia', 'GE', 0.85),
    (re.compile(r'^[A-Z]{3}\s?-?\s?\d{3}\s?-?\s?[A-Z]{2}$'), 'Georgia', 'GE', 0.8),
    
    # Armenia: 12 AA 123 (digits-letters-digits)
    (re.compile(r'^\d{2}\s?[A-Z]{2}\s?\d{3}$'), 'Armenia', 'AM', 0.9),
    
    # Azerbaijan: 12-AA-123 or 12 AA 123 (region-letters-digits)
    (re.compile(r'^\d{2}\s?-?\s?[A-Z]{2}\s?-?\s?\d{3}$'), 'Azerbaijan', 'AZ', 0.85),
    # Older Azerbaijan format: A 1234 AA
    (re.compile(r'^[A-Z]\s?\d{4}\s?[A-Z]{2}$'), 'Azerbaijan', 'AZ', 0.75),
    
    # Iran: 12 A 123 | IR 12 (2 digits, letter, 3 digits, region code)
    # OCR usually reads: 12A12345 or "12 A 123 45"
    (re.compile(r'^\d{2}\s?[A-Z]\s?\d{3}\s?\d{2}$'), 'Iran', 'IR', 0.9),
    (re.compile(r'^\d{2}[A-Z]\d{5}$'), 'Iran', 'IR', 0.85),
    
    # Iraq: 1234 A (digits + province letter) or region-specific formats
    (re.compile(r'^\d{4,5}\s?[A-Z]{1,3}$'), 'Iraq', 'IRQ', 0.7),
    (re.compile(r'^[A-Z]{1,3}\s?\d{4,5}$'), 'Iraq', 'IRQ', 0.6),
    
    # Syria: 123456 or 12-12345 (purely numeric or region-digits)
    (re.compile(r'^\d{6}$'), 'Syria', 'SYR', 0.6),
    (re.compile(r'^\d{2}\s?-?\s?\d{5}$'), 'Syria', 'SYR', 0.65),
    
    # Cyprus: ABC 123 or AAA 123 (EU member, has blue strip)
    (re.compile(r'^[A-Z]{3}\s?\d{3}$'), 'Cyprus', 'CY', 0.7),
    
    # Ukraine: AA 1234 AA (2 letters, 4 digits, 2 letters)
    (re.compile(r'^[A-Z]{2}\s?\d{4}\s?[A-Z]{2}$'), 'Ukraine', 'UA', 0.9),
    # Older: A 1234 AA
    (re.compile(r'^[A-Z]\s?\d{4}\s?[A-Z]{2}$'), 'Ukraine', 'UA', 0.8),
    
]

# EU member state codes (for narrowing "EU" to a specific country)
EU_MEMBER_CODES = {
    'A', 'B', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EST', 'FIN', 'F', 'D', 'GR',
    'H', 'IRL', 'I', 'LV', 'LT', 'L', 'M', 'NL', 'PL', 'P', 'RO', 'SK',
    'SLO', 'E', 'S',
}


class CountryIdentifier:
    """
    Identify the country of origin from a license plate.
    
    Uses multiple methods:
    1. EU blue strip detection (visual)
    2. Country code extraction from plate text
    3. Plate format pattern matching
    
    When an EU blue strip is detected, the identifier tries to narrow down
    the specific EU member state using plate format patterns and country codes.
    
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
        has_eu_strip = False
        
        # Method 1: Check for EU blue strip (visual)
        if plate_image is not None:
            eu_result = self._detect_eu_strip(plate_image)
            if eu_result:
                has_eu_strip = True
                # Keep as fallback, but don't add yet — try to find specific country first
        
        # Method 2: Country code in plate text
        if plate_text:
            code_result = self._check_country_code(plate_text)
            if code_result:
                candidates.append(code_result)
        
        # Method 3: Plate format pattern matching
        if plate_text:
            pattern_results = self._match_plate_format(plate_text)
            candidates.extend(pattern_results)
        
        # If EU strip was detected, boost confidence of any EU member state match
        if has_eu_strip and candidates:
            for c in candidates:
                if c.get('country_code', '') in EU_MEMBER_CODES:
                    # EU strip confirms it's a European plate — boost confidence
                    c['confidence'] = min(1.0, c['confidence'] + 0.15)
                    c['method'] = c['method'] + '+eu_strip'
        
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
        
        # If we only have EU strip with no specific country, use that as fallback
        if has_eu_strip:
            return {
                'country': 'European Union (member state)',
                'country_code': 'EU',
                'confidence': round(eu_result['confidence'], 4),
                'method': 'eu_blue_strip',
                'all_matches': [eu_result]
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
        Check if plate text contains a known country code token.

        Supports:
        - Prefix codes: "PL WZY54495"
        - Suffix codes: "WZY54495 PL"
        - Separated forms with hyphen/space: "TR-34ABC123"
        """
        text = plate_text.strip().upper()

        # Normalize separators and split into alphanumeric tokens.
        normalized = re.sub(r'[\-_/]+', ' ', text)
        tokens = [t for t in re.split(r'\s+', normalized) if t]

        # Check tokens first (most reliable for OCR outputs like "PL WZY54495").
        for token in tokens:
            if token in COUNTRY_CODES:
                return {
                    'country': COUNTRY_CODES[token],
                    'country_code': token,
                    'confidence': 0.9 if len(token) >= 2 else 0.75,
                    'method': 'country_code_token'
                }

        # Backward-compatible prefix fallback on compact strings.
        compact = re.sub(r'[^A-Z0-9]', '', text)
        for length in [3, 2, 1]:
            prefix = compact[:length]
            if prefix in COUNTRY_CODES:
                return {
                    'country': COUNTRY_CODES[prefix],
                    'country_code': prefix,
                    'confidence': 0.6 + (length * 0.1),
                    'method': 'country_code_prefix'
                }
        
        return None
    
    def _match_plate_format(self, plate_text):
        """
        Match plate text against known country plate formats.
        """
        text = plate_text.strip().upper()
        matches = []
        
        for pattern, country, code, confidence in PLATE_PATTERNS:
            if pattern.match(text):
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
